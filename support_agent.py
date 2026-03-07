import os
import json
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configurar el cliente de Groq (la API key se lee de GROQ_API_KEY en las vars de entorno)
client = Groq()

SYSTEM_PROMPT = """Eres el Asistente Virtual de TrailynSafe, una plataforma integral de prevención y gestión de transporte escolar.

Tu objetivo principal es brindar atención a clientes de primer nivel resolviendo dudas comunes sobre el uso de las aplicaciones (Web, App para Padres, App para Conductores y Wear OS).
Debes ser amable, conciso, respetuoso y tener un tono profesional pero cercano, similar a la IA de Mercado Pago.

INFORMACIÓN CLAVE DE TRAILYNSAFE:
1. **App Padre (Android)**: Pueden registrarse, vincular hijos (se genera QR), ver rutas disponibles, confirmar asistencia, y ver el tracking del conductor en tiempo real.
2. **App Conductor (Android)**: Pueden ver sus viajes asignados, confirmar asistencia escaneando el QR de los alumnos, ver navegación GPS, y es obligatorio que se vinculen con Wear OS.
3. **App Wear OS**: Monitorea el ritmo cardíaco (salud) del conductor en tiempo real. Obligatorio por seguridad. Si hay anomalías (ej. Taquicardia), alerta al sistema/admin.
4. **Web y Panel Admin**: Donde el administrador gestiona escuelas, padres, choferes, viajes y métricas en tiempo real.

PAUTAS DE COMPORTAMIENTO:
- Pregunta activamente cómo puedes ayudar ("¡Bienvenido a Atención a Clientes TrailynSafe! ¿En qué te puedo ayudar hoy?").
- Ve directo al grano dando pasos claros (1, 2, 3...) si el usuario pregunta cómo hacer algo.
- **NO INVENTES FUNCIONES**. Si te preguntan algo fuera de tu conocimiento, di que no estás seguro y que puedes solicitar la asistencia de un agente humano.

REGLAS DE ESCALACIÓN A AGENTE HUMANO:
- Si el usuario dice que quiere hablar con "un humano", "un agente", "una persona", o "soporte técnico real".
- Si no sabes cómo resolver el problema del usuario después de 2 intentos.
- Si el usuario tiene problemas financieros, legales o reporta fallos críticos del sistema (bugs graves).

Para escalar el chat, debes incluir exactamente y sin comillas la frase "ESCALAR_A_HUMANO" al final de tu mensaje en una nueva línea. No le digas al usuario que escribes esa frase, solamente dile, por ejemplo: "Claro, comprendo la situación. Te voy a transferir con un agente humano para que revise tu caso de inmediato. Dame un momento..." y luego incluyes la palabra clave.

EJEMPLO DE ESCALACIÓN:
Lamento mucho que sigas teniendo inconvenientes para recuperar tu contraseña. Debido a que el problema persiste, voy a transferirte a uno de nuestros agentes especializados para que libere tu cuenta de inmediato. En breve se conectará en este mismo chat.
ESCALAR_A_HUMANO
"""

def generate_chat_response(history: list[dict]):
    """
    history format: [{'role': 'user', 'content': 'hola'}, {'role': 'assistant', 'content': '...'}, ...]
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    # Validar que los roles sean correctos según la API de Groq
    for msg in history:
        # Asegurarse que el rol sea user o assistant
        role = msg.get("role")
        if role in ["user", "assistant"]:
            messages.append({"role": role, "content": msg.get("content", "")})

    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",  # Modelo rápido y eficiente en Groq
            temperature=0.7,
            max_tokens=500,
        )
        
        response_text = chat_completion.choices[0].message.content
        escalate = False
        
        # Detectar si el LLM decidió escalar
        if "ESCALAR_A_HUMANO" in response_text:
            escalate = True
            # Limpiar la bandera para no mostrársela al usuario tal cual
            response_text = response_text.replace("ESCALAR_A_HUMANO", "").strip()
            
        return {
            "success": True,
            "response": response_text,
            "escalate": escalate
        }
            
    except Exception as e:
        logger.error(f"Error calling Groq API: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def generate_chat_summary(history: list[dict]):
    """
    Genera un resumen breve del chat para que el agente humano tenga contexto rápido.
    """
    chat_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    
    summary_prompt = f"""
    Lee la siguiente conversación de soporte y genera un breve resumen de 2-3 líneas para un agente humano que va a tomar el ticket ahora.
    Dile al agente cuál es el problema exacto del usuario y qué se intentó hasta el momento.
    
    Conversación:
    {chat_text}
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Eres un asistente interno. Responde con un resumen directo, profesional e imparcial."},
                {"role": "user", "content": summary_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3, # Baja temperatura para que sea conciso y al grano
            max_tokens=200,
        )
        
        return {
            "success": True,
            "summary": chat_completion.choices[0].message.content.strip()
        }
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return {
            "success": False,
            "error": str(e)
        }
