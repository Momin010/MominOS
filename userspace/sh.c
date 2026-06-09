#include "stdio.h"
#include "string.h"
#include "syscall.h"

#define MAX_ARGS 16
#define LINE_SIZE 256
#define PATH_SIZE 256

/* Split line into argv tokens in place (whitespace separated).
   Returns argc. */
static int tokenize(char *line, char **argv) {
    int argc = 0;
    char *p = line;

    while (*p && argc < MAX_ARGS - 1) {
        while (*p == ' ' || *p == '\t' || *p == '\n')
            *p++ = 0;
        if (*p == 0)
            break;
        argv[argc++] = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n')
            p++;
    }
    argv[argc] = 0;
    return argc;
}

/* Resolve a command name to a path: try it verbatim, else /bin/<name>. */
static void resolve_cmd(const char *cmd, char *out, int cap) {
    if (cmd[0] == '/' || cmd[0] == '.') {
        int i = 0;
        while (cmd[i] && i < cap - 1) {
            out[i] = cmd[i];
            i++;
        }
        out[i] = 0;
        return;
    }

    const char *prefix = "/bin/";
    int i = 0;
    while (prefix[i]) {
        out[i] = prefix[i];
        i++;
    }
    int j = 0;
    while (cmd[j] && i < cap - 1)
        out[i++] = cmd[j++];
    out[i] = 0;
}

/* Scan argv for a `>` or `>>` redirection token. If found, set *redir_path to
   the following filename, *append accordingly, and remove both tokens from
   args (NULL-terminating at the new argc). Returns 0 on success, -1 on a
   syntax error (missing filename). No redirection -> *redir_path stays 0. */
static int parse_redirect(char **args, int *argc, const char **redir_path, int *append) {
    int i;

    *redir_path = 0;
    *append = 0;

    for (i = 0; i < *argc; i++) {
        int is_app = strcmp(args[i], ">>") == 0;
        int is_trunc = strcmp(args[i], ">") == 0;

        if (!is_app && !is_trunc)
            continue;

        if (i + 1 >= *argc)
            return -1;          /* `cmd >` with no filename */

        *append = is_app;
        *redir_path = args[i + 1];

        /* drop tokens i and i+1 by shifting the tail down */
        for (int j = i + 2; j <= *argc; j++)
            args[j - 2] = args[j];
        *argc -= 2;
        args[*argc] = 0;
        return 0;
    }

    return 0;
}

int main(int argc, char **argv) {
    char line[LINE_SIZE];
    char cwd[128];
    char *args[MAX_ARGS];
    char path[PATH_SIZE];

    (void)argc;
    (void)argv;

    printf("MominOS shell. type 'exit' to quit.\n");

    for (;;) {
        long n;
        int nargs;
        long pid;
        long code;

        syscall_getcwd(cwd, sizeof(cwd));
        printf("%s $ ", cwd);

        n = syscall_read(0, line, sizeof(line) - 1);
        if (n <= 0)
            continue;
        line[n] = 0;

        const char *redir_path;
        int append;
        struct spawn_redirect redir;

        nargs = tokenize(line, args);
        if (nargs == 0)
            continue;

        if (parse_redirect(args, &nargs, &redir_path, &append) < 0) {
            printf("sh: syntax error near redirection\n");
            continue;
        }
        if (nargs == 0) {
            printf("sh: missing command\n");
            continue;
        }

        if (strcmp(args[0], "exit") == 0)
            break;

        if (strcmp(args[0], "cd") == 0) {
            const char *target = (nargs > 1) ? args[1] : "/";
            if (syscall_chdir(target) < 0)
                printf("cd: %s: no such directory\n", target);
            continue;
        }

        redir.path = redir_path;
        redir.append = append;

        resolve_cmd(args[0], path, sizeof(path));
        pid = syscall_spawn_redir(path, args, &redir);
        if (pid < 0) {
            printf("%s: command not found\n", args[0]);
            continue;
        }

        code = syscall_waitpid(pid);
        (void)code;
    }

    printf("sh: exiting\n");
    return 0;
}
