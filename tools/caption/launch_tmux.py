import libtmux
import socket
import time

svr = libtmux.Server()

node_list = [
    "batch-block1-2057",
    "batch-block1-2008",
    "batch-block1-2119",
    "batch-block1-2145",
    "batch-block1-2035",
    "batch-block1-0049",
    "batch-block1-0002"
]

# Variable to control the number of panes per window
PANELS_PER_WINDOW = 2  # Adjust this number as needed
SESSION_NAME = 'caption'

# Check if the session already exists
existing_session = svr.find_where({"session_name": SESSION_NAME})

if existing_session:
    # Ask for user confirmation
    print(f"Session '{SESSION_NAME}' already exists.")
    print("Options:")
    print("1. Kill it and create a new one")
    print("2. Re-use the existing session and windows/panes")
    user_input = input("Please enter 1 or 2: ")

    if user_input == '1':
        # Kill the existing session
        existing_session.kill_session()
        print(f"Killed existing session '{SESSION_NAME}'")
        # Create a new session
        session = svr.new_session(session_name=SESSION_NAME)
        is_new_session = True
    elif user_input == '2':
        print(f"Re-using existing session '{SESSION_NAME}'")
        session = existing_session
        is_new_session = False
    else:
        print("Invalid input. Exiting.")
        exit()
else:
    # Create a new session if it doesn't exist
    session = svr.new_session(session_name=SESSION_NAME)
    is_new_session = True

# List to keep track of nodes and their corresponding panes
nodes_and_panes = []

# Start with the first window and pane
if is_new_session:
    window = session.attached_window
    pane_index_in_window = 0  # Counter for panes in the current window
    window_index = 0
    windows = [window]
else:
    # Re-use existing windows and panes
    windows = session.list_windows()
    window_index = 0
    window = windows[window_index]
    pane_index_in_window = 0

for idx, node in enumerate(node_list):
    # Determine whether to use existing window/pane or create new ones
    if pane_index_in_window == 0:
        if idx == 0 and is_new_session:
            # First node in a new session, use the first pane of the first window
            pane = window.panes[0]
            is_new_window = True
            is_new_pane = True
        else:
            # Create or select a new window
            window_index += 1
            if is_new_session or window_index >= len(windows):
                # Create a new window
                window = session.new_window()
                windows.append(window)
                print(f"Created new window: {window.name}")
                is_new_window = True
            else:
                # Re-use existing window
                window = windows[window_index]
                print(f"Re-using existing window: {window.name}")
                is_new_window = False
            pane = window.panes[0]
            pane_index_in_window = 0
            if is_new_window:
                is_new_pane = True
            else:
                if len(window.panes) > 0:
                    is_new_pane = False
                else:
                    is_new_pane = True
    else:
        if is_new_session or len(window.panes) <= pane_index_in_window:
            # Split the current window to create a new pane
            pane = window.split_window()
            print(f"Created new pane in window {window.name}")
            is_new_pane = True
        else:
            # Re-use existing pane
            pane = window.panes[pane_index_in_window]
            print(f"Re-using existing pane in window {window.name}")
            is_new_pane = False

    nodes_and_panes.append((node, pane))

    # Determine whether to send Ctrl+C interruption
    if not is_new_session and not is_new_window and not is_new_pane:
        # Send key interruption first
        pane.send_keys('\x03')  # Ctrl+C to interrupt any running process

    # Check if the pane is already connected to the target node
    # Send commands to get the hostname
    pane.send_keys('echo __NODE_CHECK_START__', enter=True)
    pane.send_keys('hostname', enter=True)
    pane.send_keys('echo __NODE_CHECK_END__', enter=True)
    # Wait a bit for the commands to execute
    time.sleep(0.5)
    # Capture the pane content
    content = pane.capture_pane()
    # Extract the hostname between the markers
    hostname = None
    start_found = False
    for line in content:
        if '__NODE_CHECK_START__' in line:
            start_found = True
            continue
        if '__NODE_CHECK_END__' in line:
            break
        if start_found:
            hostname = line.strip()
            break
    if hostname == node:
        print(f"Pane already connected to {node}, skipping ssh command.")
    else:
        # Not connected, need to ssh
        pane.send_keys(f'ssh {node}', enter=True)

    # Now send the rest of the commands
    pane.send_keys('cd code/CogVideo/', enter=True)
    pane.send_keys('mamba activate cogvideo', enter=True)
    pane.send_keys('torchrun --nproc-per-node=8 tools/caption/api_oci.py', enter=True)

    # Increment pane index and reset if necessary
    pane_index_in_window += 1
    if pane_index_in_window >= PANELS_PER_WINDOW:
        pane_index_in_window = 0  # Reset for the next window

# Print the IP addresses for the nodes
for node in node_list:
    ip = socket.gethostbyname(node)
    print(f'{node}: {ip}')
