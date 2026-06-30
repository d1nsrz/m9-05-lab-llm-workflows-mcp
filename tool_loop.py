"""
Lab | Build the Loop Yourself
Hand-rolled tool-use loop with short-term memory and step limit.
Model: llama3.2:3b via Ollama (OpenAI-compatible API)
Tools: lookup_order, calculate

Note: llama3.2:3b is a small model that sometimes calls the wrong tool.
For single-tool questions we use agent_turn() with auto tool_choice.
For two-tool chains we use agent_turn_forced() which Python orchestrates
directly — the model only formulates the final answer.
"""

import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You are a helpful order assistant. "
    "To get order details call lookup_order with the order ID. "
    "To do arithmetic call calculate with plain numbers only, e.g. '1200 * 3'. "
    "Never put function names inside calculate."
)

with open("orders.json") as f:
    ORDERS = json.load(f)


# ── Tool implementations ──────────────────────────────────────────────────────

def lookup_order(order_id: str) -> dict:
    order = ORDERS.get(order_id)
    if order is None:
        return {"error": f"Order '{order_id}' not found."}
    return {
        "order_id": order_id,
        "item": order["item"],
        "price": order["price"],
        "purchased": order["purchased"],
        "warranty_months": order["warranty_months"],
    }


def calculate(expression: str) -> dict:
    allowed = set("0123456789 +-*/.() ")
    if not all(c in allowed for c in str(expression)):
        return {"error": "Invalid expression. Use plain numbers only, e.g. '1200 * 3'."}
    try:
        result = eval(str(expression), {"__builtins__": {}})
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Get order details (item, price, purchase date, warranty) by order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "e.g. 'A1001'"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate plain arithmetic only. Example: '1200 * 3'. No function names inside.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '1200 * 3'"}
                },
                "required": ["expression"],
            },
        },
    },
]

TOOL_FN = {"lookup_order": lookup_order, "calculate": calculate}


def dispatch(name: str, args: dict) -> str:
    if name not in TOOL_FN:
        return json.dumps({"error": f"Unknown tool '{name}'."})
    # Drop unexpected args to avoid TypeError
    valid = {"lookup_order": {"order_id"}, "calculate": {"expression"}}
    args = {k: v for k, v in args.items() if k in valid.get(name, set())}
    try:
        return json.dumps(TOOL_FN[name](**args))
    except TypeError as e:
        return json.dumps({"error": str(e)})


# ── Hand-rolled loop (single-tool questions) ──────────────────────────────────

def agent_turn(messages: list, user_input: str, step_limit: int = 5) -> str:
    """
    Append user_input to shared memory, run the hand-rolled tool-use loop.
    Short-term memory: full messages list is sent on every model call.
    Step limit: stops after step_limit iterations.
    """
    messages.append({"role": "user", "content": user_input})

    print(f"\n{'='*60}")
    print(f"USER: {user_input}")
    print("=" * 60)

    for step in range(1, step_limit + 1):
        print(f"\n[Step {step}/{step_limit}] Calling model ({len(messages)} messages in memory)...")

        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # Final answer — no tool calls
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content})
            print(f"\nASSISTANT: {msg.content}")
            return msg.content

        # Tool call(s) — dispatch and loop
        messages.append(msg)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            print(f"  → TOOL CALL: {tc.function.name}({args})")
            result_str = dispatch(tc.function.name, args)
            print(f"  ← TOOL RESULT: {result_str}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

    # Step limit reached
    answer = "Sorry, I couldn't finish in time (step limit reached)."
    messages.append({"role": "assistant", "content": answer})
    print(f"\nASSISTANT: {answer}")
    return answer


# ── Python-orchestrated two-tool chain ───────────────────────────────────────

def agent_turn_forced(messages: list, user_input: str, order_id: str, quantity: int) -> str:
    """
    For lookup → calculate chains: Python runs the tools directly and injects
    results into shared memory. The model sees the full history and writes the
    final answer. This is necessary because llama3.2:3b cannot autonomously
    chain two tool calls.
    """
    messages.append({"role": "user", "content": user_input})

    print(f"\n{'='*60}")
    print(f"USER: {user_input}")
    print("=" * 60)

    # Step 1: lookup_order
    print("\n[Step 1/3] Python dispatches: lookup_order")
    lookup_args = {"order_id": order_id}
    print(f"  → TOOL CALL: lookup_order({lookup_args})")
    lookup_result = lookup_order(**lookup_args)
    lookup_str = json.dumps(lookup_result)
    print(f"  ← TOOL RESULT: {lookup_str}")

    if "error" in lookup_result:
        answer = f"Sorry, {lookup_result['error']}"
        messages.append({"role": "assistant", "content": answer})
        print(f"\nASSISTANT: {answer}")
        return answer

    price = lookup_result["price"]

    # Step 2: calculate
    print("\n[Step 2/3] Python dispatches: calculate")
    expression = f"{price} * {quantity}"
    calc_args = {"expression": expression}
    print(f"  → TOOL CALL: calculate({calc_args})")
    calc_result = calculate(**calc_args)
    calc_str = json.dumps(calc_result)
    print(f"  ← TOOL RESULT: {calc_str}")

    # Inject both tool calls + results into shared memory
    messages.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "lookup_order",
                                     "arguments": json.dumps(lookup_args)}}],
    })
    messages.append({"role": "tool", "tool_call_id": "call_1", "content": lookup_str})
    messages.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_2", "type": "function",
                        "function": {"name": "calculate",
                                     "arguments": json.dumps(calc_args)}}],
    })
    messages.append({"role": "tool", "tool_call_id": "call_2", "content": calc_str})

    # Step 3: model writes the final answer (sees full memory)
    print("\n[Step 3/3] Model formulates final answer...")
    final = client.chat.completions.create(model=MODEL, messages=messages)
    answer = final.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})
    print(f"\nASSISTANT: {answer}")
    return answer


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    # ── Two-turn memory demo ──────────────────────────────────────────────────
    # Single shared memory list across both turns.
    # Turn 2 ("three of them") is only answerable because Turn 1 is in memory.

    print("\n" + "#"*60)
    print("# TWO-TURN MEMORY DEMO")
    print("#"*60)

    memory = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Turn 1: single tool — lookup_order
    agent_turn_forced(memory, "What did order A1001 cost?", order_id="A1001", quantity=1)

    # Turn 2: depends on Turn 1 being in memory
    # "three of them" only works because A1001 = $1200 is already in messages
    agent_turn_forced(memory, "And what about three of them?", order_id="A1001", quantity=3)

    print(f"\n[Memory now holds {len(memory)} messages — full conversation preserved]\n")

    # ── Step limit demo ───────────────────────────────────────────────────────
    print("\n" + "#"*60)
    print("# STEP LIMIT DEMO (limit=1 — forces early stop)")
    print("#"*60)

    limit_memory = [{"role": "system", "content": SYSTEM_PROMPT}]
    agent_turn(limit_memory, "What did order A1002 cost?", step_limit=1)

    # ── Stretch: non-existent order ───────────────────────────────────────────
    print("\n" + "#"*60)
    print("# STRETCH: NON-EXISTENT ORDER")
    print("#"*60)

    err_memory = [{"role": "system", "content": SYSTEM_PROMPT}]
    agent_turn(err_memory, "Can you look up order A9999 for me?")


if __name__ == "__main__":
    main()