"""AI女友的终端入口。

用法：python call.py
"""
from chat_completion import Sister

EXIT_WORDS = {"exit", "quit", "q", "退出", "再见"}


def main():
    sister = Sister()
    print("AI女友已上线。输入 exit / quit / 退出 结束对话。")
    print()

    while True:
        try:
            user_input = input("你： ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见～")
            break

        if user_input.lower() in EXIT_WORDS:
            print("再见～")
            break
        if not user_input:
            continue

        sister.chat(user_input)
        print()


if __name__ == "__main__":
    main()
