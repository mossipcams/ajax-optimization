# Source from sayso-training/ after activating training/.venv.
# The path list is shared with pytest and the promotion CLI.
export PYTHONPATH
PYTHONPATH="$(python3 ./training_paths.py)"
