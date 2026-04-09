import json, sys, urllib.request

url = "https://starknet-sepolia.g.alchemy.com/starknet/version/rpc/v0_8/demo"
tx_hash = "0x7c5ee554baaa637fda9dfb5a92e39ef35d28e30020900d5a52778eb9615daac"

req = urllib.request.Request(url, method="POST")
req.add_header("Content-Type", "application/json")
body = json.dumps({
    "jsonrpc": "2.0", "id": 1,
    "method": "starknet_traceTransaction",
    "params": {"transaction_hash": tx_hash}
}).encode()

resp = urllib.request.urlopen(req, body)
data = json.loads(resp.read())
result = data.get("result", {})
ei = result.get("execute_invocation", {})

def decode_hex(h):
    clean = h[2:].lstrip("0") or "0"
    if len(clean) % 2:
        clean = "0" + clean
    try:
        b = bytes.fromhex(clean)
        t = b.decode("utf-8", errors="ignore")
        t = "".join(c for c in t if 32 <= ord(c) < 127)
        if len(t) > 3:
            return t
    except:
        pass
    return None

def walk(obj, depth=0):
    if isinstance(obj, dict):
        rr = obj.get("revert_reason", "")
        if rr:
            indent = "  " * depth
            print(f"{indent}REVERT: {rr[:200]}")
            # Decode hex in revert
            import re
            for h in re.findall(r"0x[0-9a-fA-F]{8,60}", rr):
                d = decode_hex(h)
                if d:
                    print(f"{indent}  -> {d}")
        for k in ["calls"]:
            if k in obj:
                for i, c in enumerate(obj[k]):
                    sel = c.get("entry_point_selector", "")
                    ct = c.get("contract_address", "")
                    print(f"{'  '*depth}[{i}] contract={ct[:20]}... selector={sel[:20]}...")
                    walk(c, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            walk(item, depth)

walk(ei)
