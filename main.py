import sys

from combined_foundation.cli import cli


def interactive_menu():
    print("===================================================")
    print(" Obscuro Erebus Combined Foundation Model CLI")
    print("===================================================")
    print("Select a command to run:")
    print(" [1] demo      - Architecture demo & model printout")
    print(" [2] verify    - Verify architecture support")
    print(" [3] train     - Synthetic RL training cycle")
    print(" [4] industrial- Multimodal RL pipeline")
    print(" [5] fullrun   - Full industrial runtime execution")
    print(" [6] exit")
    print()

    cmd_map = {
        "1": "demo",
        "2": "verify",
        "3": "train",
        "4": "industrial",
        "5": "fullrun",
    }

    try:
        choice = input("Enter choice (1-6): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if choice in ("6", "exit", "q", "quit"):
        return

    selected_cmd = cmd_map.get(choice, choice)
    print(f"\n---> Running: {selected_cmd}\n")
    try:
        cli.main(args=[selected_cmd], prog_name="combined_foundation_model", standalone_mode=False)
    except Exception as e:
        print(f"Error executing command '{selected_cmd}': {e}")

    try:
        input("\nPress Enter to exit...")
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    if len(sys.argv) == 1:
        try:
            interactive_menu()
        except (EOFError, KeyboardInterrupt):
            pass
    else:
        cli()
