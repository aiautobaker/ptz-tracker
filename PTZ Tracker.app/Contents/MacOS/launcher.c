#include <mach-o/dyld.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char exe_path[PATH_MAX];
    uint32_t size = (uint32_t)sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) != 0) {
        fprintf(stderr, "Failed to resolve executable path.\n");
        return 1;
    }

    char resolved_path[PATH_MAX];
    if (realpath(exe_path, resolved_path) == NULL) {
        perror("realpath");
        return 1;
    }

    char *last_slash = strrchr(resolved_path, '/');
    if (last_slash == NULL) {
        fprintf(stderr, "Invalid executable path.\n");
        return 1;
    }
    *last_slash = '\0';

    char launch_script[PATH_MAX];
    int written = snprintf(launch_script, sizeof(launch_script), "%s/launch.sh", resolved_path);
    if (written <= 0 || (size_t)written >= sizeof(launch_script)) {
        fprintf(stderr, "Failed to build launcher script path.\n");
        return 1;
    }

    char *const argv[] = {"/bin/bash", launch_script, NULL};
    execv("/bin/bash", argv);

    perror("execv");
    return 1;
}
