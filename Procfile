web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120

# Heroku/Platform: set OPENAI_API_KEY as a config var (do not commit the key)
# Example:
#   heroku config:set OPENAI_API_KEY="sk-..."
