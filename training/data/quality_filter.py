#!/usr/bin/env python3
"""Quality filter for synthetic kernel fault diagnosis dataset.

Rejects low-quality SFT/DPO samples based on:
  - Minimum length requirements
  - Content quality heuristics (specificity, correctness indicators)
  - Blacklisted phrases (vague, non-actionable responses)
  - Keyword coverage against expected patterns
"""

import re
from typing import List, Dict, Tuple, Optional


# 📏 Length thresholds
MIN_SFT_RESPONSE_LENGTH = 80         # Minimum chars for an SFT response
MIN_DPO_CHOSEN_LENGTH = 80           # Minimum chars for DPO chosen
MIN_DPO_REJECTED_LENGTH = 30         # Minimum chars for DPO rejected
MAX_SFT_RESPONSE_LENGTH = 8000       # Maximum chars (truncated responses)
MIN_PROMPT_LENGTH = 20               # Minimum chars for a prompt

# 🚫 Blacklisted phrases — indicate low-quality or AI-hallucinated content
BLACKLISTED_PHRASES = [
    "I am an AI", "as an AI", "I cannot", "I'm unable to",
    "I don't have access", "I do not have access",
    "it is important to note that", "please note that",
    "in conclusion", "in summary", "to summarize",
    "I would recommend", "I suggest you", "you should consider",
    "as previously mentioned", "as mentioned earlier",
    "it depends", "it varies", "it could be",
    "one possible cause", "there could be many reasons",
    "this is a complex issue", "this is a broad topic",
    "further investigation is needed", "more research is needed",
    "consult the documentation", "refer to the manual",
    "contact support", "reach out to",
    "depending on the situation", "depending on the context",
    "typically", "usually", "often",
    "Lorem ipsum",
]

# ✅ Good keyword patterns for kernel diagnosis
STRONG_KEYWORDS = [
    # Kernel mechanisms
    r"\boom[-_]?killer\b", r"\bOOM\b", r"\bcgroup\b", r"\bslab\b",
    r"\bdentry\b", r"\binode\b", r"\bRCU\b", r"\bpreempt\b",
    r"\bspinlock\b", r"\bmutex\b", r"\bsemaphore\b",
    r"\bIRQ\b", r"\binterrupt\b", r"\bNAPI\b",
    r"\bDMA\b", r"\bIOMMU\b", r"\bMSI[-_]?X\b",
    r"\bNUMA\b", r"\bnumactl\b", r"\bpage fault\b",
    r"\bkmalloc\b", r"\bvmalloc\b", r"\bkfree\b",
    r"\bkmemleak\b", r"\bkasan\b", r"\bUAF\b",
    r"\bTOCTOU\b", r"\bSELinux\b", r"\bAppArmor\b",
    r"\bfsck\b", r"\bext[234]\b", r"\bjournal\b",
    r"\bworkqueue\b", r"\bflush\b", r"\bGPF\b",
    r"\bNULL pointer\b", r"\bdereference\b",
    r"\bmodprobe\b", r"\brmmod\b", r"\binsmod\b",
    r"\btcp_mem\b", r"\bcfs_quota\b", r"\bio_uring\b",
    r"\bTTM\b", r"\bVRAM\b", r"\bGPU hang\b",
    # Action words
    r"\bfix\b", r"\bpatch\b", r"\bworkaround\b",
    r"\bdiagnos[ei]\w*\b", r"\broot cause\b",
    r"\b[sys]?ctl\b", r"\bdebugfs\b",
]

WEAK_PHRASES = [
    "add more RAM", "add more memory", "upgrade hardware",
    "replace hardware", "reboot", "restart", "reinstall",
    "reformat", "format and restore", "upgrade CPU",
    "faster processor", "upgrade storage",
    "update kernel", "upgrade kernel", "newer kernel",
    "switch to", "try using", "try a different",
    "check logs", "check the logs", "examine logs",
]


class QualityFilter:
    """Filter and score synthetic data quality."""
    
    def __init__(self):
        self.blacklisted = [p.lower() for p in BLACKLISTED_PHRASES]
        self.strong_patterns = [re.compile(p, re.IGNORECASE) for p in STRONG_KEYWORDS]
        self.weak_patterns = [re.compile(p, re.IGNORECASE) for p in WEAK_PHRASES]
    
    def _check_blacklist(self, text: str) -> List[str]:
        """Check for blacklisted phrases. Returns list of violations."""
        text_lower = text.lower()
        found = []
        for phrase in self.blacklisted:
            if phrase in text_lower:
                found.append(phrase)
        return found
    
    def _check_strong_keywords(self, text: str) -> int:
        """Count strong keyword matches."""
        count = 0
        for pattern in self.strong_patterns:
            if pattern.search(text):
                count += 1
        return count
    
    def _check_weak_phrases(self, text: str) -> int:
        """Count weak/superficial phrase matches."""
        count = 0
        for pattern in self.weak_patterns:
            if pattern.search(text):
                count += 1
        return count
    
    def _check_expected_patterns(self, text: str, expected: List[str]) -> Tuple[int, int]:
        """Check coverage of expected patterns. Returns (matched, total)."""
        matched = 0
        for pattern_str in expected:
            pattern = re.compile(re.escape(pattern_str), re.IGNORECASE)
            if pattern.search(text):
                matched += 1
        return matched, len(expected)
    
    def evaluate_sft(self, sample: Dict) -> Dict:
        """Evaluate a single SFT sample. Returns quality report dict."""
        prompt = sample.get("prompt", "")
        response = sample.get("response", "")
        metadata = sample.get("metadata", {})
        expected_patterns = metadata.get("expected_patterns", [])
        
        fail_reasons = []
        warnings = []
        
        # Length checks
        if len(response) < MIN_SFT_RESPONSE_LENGTH:
            fail_reasons.append(f"response too short ({len(response)} < {MIN_SFT_RESPONSE_LENGTH})")
        if len(response) > MAX_SFT_RESPONSE_LENGTH:
            fail_reasons.append(f"response too long ({len(response)} > {MAX_SFT_RESPONSE_LENGTH})")
        if len(prompt) < MIN_PROMPT_LENGTH:
            fail_reasons.append(f"prompt too short ({len(prompt)} chars)")
        
        # Blacklist check
        blacklisted = self._check_blacklist(response)
        if blacklisted:
            fail_reasons.append(f"blacklisted phrases: {blacklisted[:3]}")
        
        # Keyword strength
        strong_count = self._check_strong_keywords(response)
        if strong_count < 2:
            fail_reasons.append(f"too few strong kernel keywords ({strong_count})")
        
        # Weak phrases
        weak_count = self._check_weak_phrases(response)
        if weak_count >= 3:
            warnings.append(f"high weak phrase count ({weak_count})")
        
        # Expected pattern coverage
        if expected_patterns:
            matched, total = self._check_expected_patterns(response, expected_patterns)
            coverage = matched / total if total > 0 else 1.0
            if coverage < 0.5:
                fail_reasons.append(f"low expected pattern coverage ({matched}/{total})")
        else:
            coverage = 1.0
        
        # Heuristic: response should contain some action/fix suggestion
        has_action = any(
            word in response.lower()
            for word in ["fix", "set ", "use ", "add ", "enable", "disable", "run ", "check "]
        )
        if not has_action:
            warnings.append("no actionable suggestion found")
        
        passed = len(fail_reasons) == 0
        
        return {
            "pass": passed,
            "score": strong_count - weak_count,
            "fail_reasons": fail_reasons,
            "warnings": warnings,
            "length": len(response),
            "strong_keywords": strong_count,
            "weak_phrases": weak_count,
            "pattern_coverage": coverage,
        }
    
    def evaluate_dpo(self, sample: Dict) -> Dict:
        """Evaluate a DPO pair. The chosen must be higher quality than rejected."""
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")
        metadata = sample.get("metadata", {})
        expected_patterns = metadata.get("expected_patterns", [])
        
        fail_reasons = []
        warnings = []
        
        # Length checks
        if len(chosen) < MIN_DPO_CHOSEN_LENGTH:
            fail_reasons.append(f"chosen too short ({len(chosen)} < {MIN_DPO_CHOSEN_LENGTH})")
        if len(rejected) < MIN_DPO_REJECTED_LENGTH:
            warnings.append(f"rejected very short ({len(rejected)} chars)")
        
        # Blacklist check (chosen should not have blacklisted phrases)
        blacklisted_chosen = self._check_blacklist(chosen)
        if blacklisted_chosen:
            fail_reasons.append(f"chosen has blacklisted phrases: {blacklisted_chosen[:3]}")
        
        blacklisted_rej = self._check_blacklist(rejected)
        if blacklisted_rej:
            warnings.append(f"rejected has blacklisted phrases: {blacklisted_rej[:3]}")
        
        # Strong keywords: chosen must have more than rejected
        chosen_strong = self._check_strong_keywords(chosen)
        rejected_strong = self._check_strong_keywords(rejected)
        if chosen_strong <= rejected_strong:
            fail_reasons.append(
                f"chosen not sufficiently stronger than rejected "
                f"({chosen_strong} vs {rejected_strong} keywords)"
            )
        
        # Weak phrases: rejected should have more than chosen
        chosen_weak = self._check_weak_phrases(chosen)
        rejected_weak = self._check_weak_phrases(rejected)
        if chosen_weak > rejected_weak and chosen_weak > 1:
            fail_reasons.append(
                f"chosen has more weak phrases than rejected "
                f"({chosen_weak} vs {rejected_weak})"
            )
        
        # Expected patterns: chosen should cover them
        if expected_patterns:
            matched, total = self._check_expected_patterns(chosen, expected_patterns)
            coverage = matched / total if total > 0 else 1.0
            if coverage < 0.4:
                fail_reasons.append(f"chosen low pattern coverage ({matched}/{total})")
        else:
            coverage = 1.0
        
        # Chosen should be longer (more detailed) than rejected
        if len(chosen) < len(rejected) * 0.8:
            warnings.append(
                f"chosen ({len(chosen)}) not much longer than rejected ({len(rejected)})"
            )
        
        passed = len(fail_reasons) == 0
        
        return {
            "pass": passed,
            "score": chosen_strong - rejected_strong,
            "fail_reasons": fail_reasons,
            "warnings": warnings,
            "chosen_length": len(chosen),
            "rejected_length": len(rejected),
            "chosen_strong_keywords": chosen_strong,
            "rejected_strong_keywords": rejected_strong,
            "chosen_weak_phrases": chosen_weak,
            "rejected_weak_phrases": rejected_weak,
            "pattern_coverage": coverage,
        }
    
    def evaluate(self, sample: Dict) -> Dict:
        """Auto-detect SFT vs DPO format and evaluate."""
        if "chosen" in sample and "rejected" in sample:
            return self.evaluate_dpo(sample)
        return self.evaluate_sft(sample)
    
    def batch_evaluate(self, samples: List[Dict]) -> List[Dict]:
        """Evaluate a list of samples. Returns list of quality reports."""
        return [self.evaluate(s) for s in samples]
    
    def filter_batch(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Filter samples, return (passed, rejected) lists."""
        passed = []
        rejected = []
        for s in samples:
            quality = self.evaluate(s)
            if quality["pass"]:
                s["quality"] = quality
                passed.append(s)
            else:
                rejected.append((s, quality))
        return passed, rejected


def print_quality_summary(samples: List[Dict], label: str = ""):
    """Print a quality summary for a list of samples."""
    if not samples:
        print(f"  {label}: [no samples]")
        return
    
    scores = [s.get("quality", {}).get("score", 0) for s in samples]
    lengths = [len(s.get("response", s.get("chosen", ""))) for s in samples]
    
    print(f"  {label}: {len(samples)} samples, "
          f"avg score={sum(scores)/len(scores):.1f}, "
          f"avg length={sum(lengths)/len(lengths):.0f} chars")