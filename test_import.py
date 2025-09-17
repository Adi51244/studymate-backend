try:
    import main
    print("Successfully imported main module")
    print("Supabase URL:", main.supabase_url[:50] + "...")
    print("FastAPI app:", main.app)
except Exception as e:
    print(f"Error importing main: {e}")
    import traceback
    traceback.print_exc()