# Round 0 GitHub Upload Note

The first push to `bds299792458/openviking` was rejected by GitHub because the current HTTPS credential does not have the `workflow` scope required to create or update `.github/workflows/*`.

To keep progress moving without changing credentials, the upload branch removes GitHub Actions workflow files only. Source code, benchmark scripts, documentation, and experiment-analysis notes remain included. External datasets are still kept outside git.

If workflow files are needed later, push them with a token that has the GitHub `workflow` scope.
