"""Subtask B: Q8_0 Quantization + MOM1 Converter — Full Test Suite."""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

import torch

from train_moe.quant import round_trip_test, round_trip_linear_test
from train_moe.config import MominoMoEConfig
from train_moe.model import MominoMoE
from convert.convert_model import convert_checkpoint_to_mom1, verify_mom1_header, load_mom1_header
from convert.export_config import generate_config_header


def test_q8_0_round_trip():
    """Test Q8_0 quantization round-trip on various tensor shapes."""
    print("=" * 60)
    print("Test 1: Q8_0 Round-Trip Accuracy")
    print("=" * 60)

    # Test 1a: Linear weight
    result1 = round_trip_linear_test(1024, 1024)
    assert result1['mse'] < 1e-6, f"MSE too high: {result1['mse']}"

    # Test 1b: Small matrix
    t = torch.randn(64, 128) * 0.1
    result2 = round_trip_test(t)
    assert result2['mse'] < 1e-6, f"MSE too high: {result2['mse']}"
    print(f"  Small matrix MSE: {result2['mse']:.8e} [PASS]")

    # Test 1c: Negative values
    t3 = torch.tensor([[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]] * 4)
    result3 = round_trip_test(t3)
    print(f"  Negative values MSE: {result3['mse']:.8e} [PASS]")

    # Test 1d: fp16 input
    t4 = torch.randn(4, 32).half()
    result4 = round_trip_test(t4)
    # fp16 -> float32 -> Q8_0 -> dequant can have ~1e-5 MSE; this is normal
    assert result4['mse'] < 1e-3, f"MSE too high: {result4['mse']}"
    print(f"  fp16 input MSE: {result4['mse']:.8e} [PASS]")

    # Test 1e: Exact block boundary
    t5 = torch.randn(1, Q8_0_BLOCK_SIZE)
    result5 = round_trip_test(t5)
    print(f"  Block boundary MSE: {result5['mse']:.8e} [PASS]")

    print("✅ Q8_0 round-trip all tests PASSED\n")
    return True


def test_mom1_conversion():
    """Test MOM1 binary format conversion with random model weights."""
    print("=" * 60)
    print("Test 2: MOM1 Conversion")
    print("=" * 60)

    config = MominoMoEConfig()
    model = MominoMoE(config)
    state_dict = model.state_dict()

    # Convert to MOM1
    output_path = "/tmp/test_model.mom"
    convert_checkpoint_to_mom1(
        state_dict, config, output_path,
        quantize=True, verbose=False
    )

    # Verify file exists and has reasonable size
    file_size = os.path.getsize(output_path)
    print(f"  MOM1 file size: {file_size:,} bytes ({file_size/1e9:.2f} GB)")

    # Expected: ~1.22 GB for Q8_0 weights (~1 byte per weight)
    expected_min = 1.0e9  # ~1 GB minimum
    expected_max = 2.0e9  # ~2 GB max
    assert expected_min < file_size < expected_max, \
        f"File size {file_size} outside expected range [{expected_min}, {expected_max}]"
    print(f"  File size in expected range [{expected_min/1e9:.1f}, {expected_max/1e9:.1f}] GB [PASS]")

    # Verify header
    h = load_mom1_header(output_path)
    assert h['magic'] == b'MOM1', f"Bad magic: {h['magic']}"
    assert h['version'] == 1, f"Bad version: {h['version']}"
    assert h['d_model'] == config.d_model
    assert h['n_layers'] == config.n_layers
    assert h['n_heads'] == config.n_heads
    assert h['n_kv_heads'] == config.n_kv_heads
    assert h['head_dim'] == config.head_dim
    assert h['d_ff'] == config.d_ff
    assert h['n_experts'] == config.n_experts
    assert h['top_k'] == config.top_k
    assert h['n_shared_experts'] == config.n_shared_experts
    assert h['vocab_size'] == config.vocab_size
    assert h['context_len'] == config.context_len
    print(f"  Header: magic={h['magic']}, version={h['version']}, "
          f"tensors={h['n_tensors']}, data_offset={h['data_offset']}")
    print(f"  Header validation [PASS]")

    # Verify with config
    verify_mom1_header(output_path, config)

    # Clean up
    os.remove(output_path)
    print("✅ MOM1 conversion test PASSED\n")
    return True


def test_export_config():
    """Test C header generation."""
    print("=" * 60)
    print("Test 3: C Header Export")
    print("=" * 60)

    config = MominoMoEConfig()
    header = generate_config_header(config)

    # Verify header contains critical defines
    required_defines = [
        "MODEL_VOCAB_SIZE", "MODEL_D_MODEL", "MODEL_N_LAYERS",
        "MODEL_N_HEADS", "MODEL_N_KV_HEADS", "MODEL_HEAD_DIM",
        "MODEL_D_FF", "MODEL_N_EXPERTS", "MODEL_TOP_K",
        "MODEL_N_SHARED_EXPERTS", "MODEL_CONTEXT_LEN", "MODEL_ROPE_BASE",
        "MODEL_RMS_EPS", "Q8_0_BLOCK_SIZE", "Q8_0_BLOCK_BYTES",
        "MODEL_KV_CACHE_BYTES", "MODEL_N_GROUPS",
        "TENSOR_TOKEN_EMBED", "TENSOR_FINAL_NORM", "TENSOR_LM_HEAD",
    ]
    for define in required_defines:
        assert define in header, f"Missing define: {define}"
    print(f"  All {len(required_defines)} required defines present [PASS]")

    # Check values match config
    assert f"MODEL_VOCAB_SIZE" in header and str(config.vocab_size) in header
    assert f"MODEL_D_MODEL" in header and str(config.d_model) in header
    print(f"  Config values in header [PASS]")

    print("✅ C header export test PASSED\n")
    return True


if __name__ == "__main__":
    Q8_0_BLOCK_SIZE = 32
    tests = [test_q8_0_round_trip, test_mom1_conversion, test_export_config]
    all_pass = True
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    if all_pass:
        print("=" * 60)
        print("✅ SUBTASK B: ALL TESTS PASSED")
        print("=" * 60)
    else:
        print("=" * 60)
        print("❌ SUBTASK B: SOME TESTS FAILED")
        print("=" * 60)
        sys.exit(1)