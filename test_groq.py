import os

# Set variable explicitly for the test
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")

from support_agent import generate_chat_response, generate_chat_summary

print("==== PRUEBA DE CHAT ====")
history = [{"role": "user", "content": "Hola, ¿cómo puedo recuperar la contraseña de mi cuenta de conductor?"}]
res = generate_chat_response(history)
print(res)

print("\n==== PRUEBA DE ESCALACIÓN ====")
history.append({"role": "assistant", "content": res.get("response", "")})
history.append({"role": "user", "content": "No entiendo nada de lo que dices, quiero hablar con un humano por favor"})
res2 = generate_chat_response(history)
print(res2)

print("\n==== PRUEBA DE RESUMEN ====")
history.append({"role": "assistant", "content": res2.get("response", "")})
res_sum = generate_chat_summary(history)
print(res_sum)
