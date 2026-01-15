class Sandbox:
    def __init__(self):
        # Later: initialize Docker client, image, etc.
        ...

    async def run_python(self, code: str):
        # Later: run code in container.
        # For now, just eval in-process (unsafe, but fine for early dev).
        # DO NOT SHIP THIS.
        local_vars = {}
        try:
            exec(code, {}, local_vars)
            return {"output": local_vars}
        except Exception as e:
            return {"error": str(e)}
