from flask import Flask, jsonify, request

print(">>> app.py importado correctamente", flush=True)

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    print(">>> Entró a /health", flush=True)
    return jsonify({"status": "ok"}), 200

@app.route("/api/generar-ruta", methods=["POST"])
def generar_ruta():
    data = request.get_json(silent=True) or {}
    print(">>> Body recibido en /api/generar-ruta:", data, flush=True)

    return jsonify({
        "success": True,
        "echo": data
    }), 200

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(">>> Levantando Flask en puerto", port, flush=True)
    app.run(host="0.0.0.0", port=port)
