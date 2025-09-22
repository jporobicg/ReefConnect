#!/usr/bin/env python3
"""
Update TODO list status for connectivity matrix comparison tasks
"""

import os
import re
import sys
from datetime import datetime

def update_task_status(todo_file, task_pattern, new_status):
    """Update a specific task's status in the TODO file"""
    
    if not os.path.exists(todo_file):
        print(f"❌ TODO file not found: {todo_file}")
        return False
    
    # Read the file
    with open(todo_file, 'r') as f:
        lines = f.readlines()
    
    # Find and update the task
    updated = False
    for i, line in enumerate(lines):
        if task_pattern.lower() in line.lower() and line.strip().startswith('- ['):
            # Extract current status and task text
            match = re.match(r'^- \[([^\]]*)\](.*)$', line.strip())
            if match:
                current_status = match.group(1)
                task_text = match.group(2).strip()
                
                # Update the status
                new_line = f"- [{new_status}]{task_text}\n"
                lines[i] = new_line
                updated = True
                
                print(f"✅ Updated task: {task_text}")
                print(f"   Status: {current_status} → {new_status}")
                break
    
    if not updated:
        print(f"❌ Task not found: {task_pattern}")
        return False
    
    # Write the updated file
    with open(todo_file, 'w') as f:
        f.writelines(lines)
    
    # Add update timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"📝 Updated {todo_file} at {timestamp}")
    
    return True

def list_available_tasks(todo_file):
    """List available tasks for updating"""
    if not os.path.exists(todo_file):
        print(f"❌ TODO file not found: {todo_file}")
        return
    
    print("📋 AVAILABLE TASKS")
    print("=" * 50)
    
    with open(todo_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith('- [') and ']' in line:
                status = line[2:line.index(']')]
                task_text = line[line.index(']') + 1:].strip()
                
                # Clean up task text for display
                display_text = task_text.replace('**', '').strip()
                if len(display_text) > 60:
                    display_text = display_text[:57] + "..."
                
                print(f"{line_num:3d}. [{status}] {display_text}")

def main():
    """Main function"""
    todo_file = 'docs/connectivity_comparison_todo.md'
    
    if len(sys.argv) < 2:
        print("Usage: python update_todo_status.py <task_pattern> <new_status>")
        print("       python update_todo_status.py --list")
        print()
        print("Status options: ✅ (completed), 🔄 (in progress), ⏸️ (paused), ❌ (cancelled)")
        print()
        list_available_tasks(todo_file)
        return
    
    if sys.argv[1] == '--list':
        list_available_tasks(todo_file)
        return
    
    if len(sys.argv) < 3:
        print("❌ Please provide both task pattern and new status")
        print("Usage: python update_todo_status.py <task_pattern> <new_status>")
        return
    
    task_pattern = sys.argv[1]
    new_status = sys.argv[2]
    
    # Validate status
    valid_statuses = ['✅', '🔄', '⏸️', '❌', '']
    if new_status not in valid_statuses:
        print(f"❌ Invalid status: {new_status}")
        print(f"Valid statuses: {', '.join(valid_statuses)}")
        return
    
    # Update the task
    success = update_task_status(todo_file, task_pattern, new_status)
    
    if success:
        print("🎉 Task status updated successfully!")
    else:
        print("❌ Failed to update task status")

if __name__ == "__main__":
    main()
