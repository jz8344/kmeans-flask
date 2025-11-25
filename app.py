from flask import Flask, jsonify, request
import os

print(">>> app.py importado correctamente", flush=True)

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    print(">>> Entró a /health", flush=True)
    return jsonify({"status": "ok"}), 200

@app.route("/api/generar-ruta", methods=["GET", "POST"])
def generar_ruta():
    if request.method == "GET":
        print(">>> Entró a /api/generar-ruta con GET", flush=True)
        return jsonify({"status": "ok", "method": "GET"}), 200

    data = request.get_json(silent=True) or {}
    print(">>> Body recibido en /api/generar-ruta:", data, flush=True)

    return jsonify({
        "success": True,
        "echo": data
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(">>> Levantando Flask en puerto", port, flush=True)
    app.run(host="0.0.0.0", port=port)
