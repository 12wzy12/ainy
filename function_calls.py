from datetime import datetime


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "根据用户说的地点获取到该地的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "用户提供的地点，比如北京、上海等",
                    }
                },
                "required": ["location"]
                }
        },

    },
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "获取今天日期",
            "parameters":{
                "type": "object",
            }
        }
    }
]


def _get_weather(location):
    return f"{location}天气良好,23℃."

def _get_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")