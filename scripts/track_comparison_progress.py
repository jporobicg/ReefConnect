#!/usr/bin/env python3
"""
Track progress on connectivity matrix comparison TODO list
"""

import os
import re
from datetime import datetime

def parse_todo_file(todo_file):
    """Parse the TODO file and extract task information"""
    tasks = []
    current_section = None
    current_subsection = None
    
    with open(todo_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Check for section headers
            if line.startswith('## '):
                current_section = line[3:].strip()
                current_subsection = None
            elif line.startswith('### '):
                current_subsection = line[4:].strip()
            
            # Check for tasks
            elif line.startswith('- [') and ']' in line:
                status = line[2:line.index(']')]
                task_text = line[line.index(']') + 1:].strip()
                
                # Determine task type
                task_type = 'task'
                if task_text.startswith('**'):
                    task_type = 'major_task'
                elif task_text.startswith('  - ['):
                    task_type = 'subtask'
                
                tasks.append({
                    'line_number': line_num,
                    'section': current_section,
                    'subsection': current_subsection,
                    'status': status,
                    'text': task_text,
                    'type': task_type
                })
    
    return tasks

def count_tasks_by_status(tasks):
    """Count tasks by status"""
    status_counts = {}
    for task in tasks:
        status = task['status']
        if status not in status_counts:
            status_counts[status] = 0
        status_counts[status] += 1
    return status_counts

def count_tasks_by_section(tasks):
    """Count tasks by section"""
    section_counts = {}
    for task in tasks:
        section = task['section']
        if section:
            if section not in section_counts:
                section_counts[section] = {'total': 0, 'completed': 0, 'in_progress': 0}
            section_counts[section]['total'] += 1
            if task['status'] == '✅':
                section_counts[section]['completed'] += 1
            elif task['status'] == '🔄':
                section_counts[section]['in_progress'] += 1
    return section_counts

def generate_progress_report(todo_file):
    """Generate a progress report"""
    print("="*60)
    print("CONNECTIVITY COMPARISON PROGRESS REPORT")
    print("="*60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Parse tasks
    tasks = parse_todo_file(todo_file)
    
    # Count by status
    status_counts = count_tasks_by_status(tasks)
    total_tasks = len(tasks)
    
    print("📊 OVERALL PROGRESS")
    print("-" * 30)
    print(f"Total tasks: {total_tasks}")
    print()
    
    for status, count in sorted(status_counts.items()):
        percentage = (count / total_tasks) * 100
        status_name = {
            '✅': 'Completed',
            '🔄': 'In Progress', 
            '⏸️': 'Paused',
            '❌': 'Cancelled',
            '': 'Not Started'
        }.get(status, status)
        
        print(f"{status} {status_name}: {count} ({percentage:.1f}%)")
    
    print()
    
    # Count by section
    section_counts = count_tasks_by_section(tasks)
    
    print("📋 PROGRESS BY SECTION")
    print("-" * 30)
    for section, counts in section_counts.items():
        if section:
            completed_pct = (counts['completed'] / counts['total']) * 100
            in_progress_pct = (counts['in_progress'] / counts['total']) * 100
            
            print(f"\n{section}:")
            print(f"  Total: {counts['total']}")
            print(f"  Completed: {counts['completed']} ({completed_pct:.1f}%)")
            print(f"  In Progress: {counts['in_progress']} ({in_progress_pct:.1f}%)")
    
    print()
    
    # Recent completions (if any)
    completed_tasks = [t for t in tasks if t['status'] == '✅']
    if completed_tasks:
        print("✅ RECENTLY COMPLETED TASKS")
        print("-" * 30)
        for task in completed_tasks[-5:]:  # Show last 5
            print(f"  • {task['text']}")
        print()
    
    # Next priorities
    not_started = [t for t in tasks if t['status'] == '' and t['type'] == 'major_task']
    in_progress = [t for t in tasks if t['status'] == '🔄']
    
    if in_progress:
        print("🔄 CURRENTLY IN PROGRESS")
        print("-" * 30)
        for task in in_progress:
            print(f"  • {task['text']}")
        print()
    
    if not_started:
        print("📋 NEXT PRIORITIES")
        print("-" * 30)
        for task in not_started[:5]:  # Show first 5
            print(f"  • {task['text']}")
        print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 30)
    
    if status_counts.get('', 0) > total_tasks * 0.8:
        print("  • Focus on starting high-priority tasks")
        print("  • Consider breaking down large tasks into smaller ones")
    elif status_counts.get('🔄', 0) > 5:
        print("  • Consider completing in-progress tasks before starting new ones")
        print("  • Review task priorities and dependencies")
    elif status_counts.get('✅', 0) > total_tasks * 0.5:
        print("  • Great progress! Continue with medium-priority tasks")
        print("  • Consider starting documentation tasks")
    else:
        print("  • Steady progress - continue with current approach")
        print("  • Focus on completing immediate tasks first")

def main():
    """Main function"""
    todo_file = 'docs/connectivity_comparison_todo.md'
    
    if not os.path.exists(todo_file):
        print(f"❌ TODO file not found: {todo_file}")
        return
    
    generate_progress_report(todo_file)

if __name__ == "__main__":
    main()
