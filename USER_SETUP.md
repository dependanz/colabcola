# Colabcola user setup

The complete illustrated instructions now live in
[README: user-owned project setup](README.md#user-owned-project-setup).

That is the canonical walkthrough for installation, terminal login, Google
Console configuration, account authorization, packaging, running and recovery.

For a fresh login, use a new, unused profile name:

```bash
colabcola login --profile danzel --status
# Expect state: not_configured for an unused name.
colabcola login --profile danzel
```

Named profiles live at `~/.colabcola/configs/NAME`. Existing logins are kept in
place, and an existing name is reused; `--force` does not reset it. Use the same
profile for execution (`colabcola --profile danzel JOB.colajob` or
`colabcola run --profile danzel ...`) and recovery. See
[profile selection and existing logins](README.md#profile-selection-and-existing-logins)
for session defaults, custom directories and compatibility.
