// Storage key for localStorage
const STORAGE_KEY = 'todo-app-tasks';

// Task list references
let taskListFiltered = [];

// Select DOM elements
const taskInput = document.getElementById('taskInput');
const addBtn = document.getElementById('addBtn');
const taskList = document.getElementById('taskList');
const filterBtns = document.querySelectorAll('.filter-btn');
const remainingCount = document.getElementById('remainingCount');
const clearCompletedBtn = document.getElementById('clearCompletedBtn');

// Initialize the app
function init() {
    loadTasks();
    render();
    applyFilter('all');
    setupEventListeners();
}

// Setup all event listeners
function setupEventListeners() {
    // Add task on button click
    addBtn.addEventListener('click', () => {
        const task = taskInput.value.trim();
        if (task) {
            addTask(task);
            taskInput.value = '';
        }
    });

    // Add task on Enter key
    taskInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const task = taskInput.value.trim();
            if (task) {
                addTask(task);
                taskInput.value = '';
            }
        }
    });

    // Filter buttons
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => applyFilter(btn.dataset.filter));
    });

    // Clear completed button
    clearCompletedBtn.addEventListener('click', clearCompletedTasks);

    // Double-click to edit task text
    taskList.addEventListener('dblclick', (e) => {
        const taskText = e.target.closest('.task-text');
        if (taskText) {
            const taskId = parseInt(taskText.dataset.id);
            editTask(taskId);
        }
    });
}

// Add a new task
function addTask(text) {
    const task = {
        id: Date.now(),
        text: text,
        completed: false,
        createdAt: Date.now()
    };
    const tasks = getTasks();
    tasks.push(task);
    saveTasks(tasks);
    render();
}

// Edit a task text inline
function editTask(id) {
    const task = getTaskById(id);
    if (!task) return;

    // Create editable span
    const span = document.createElement('span');
    span.textContent = task.text;
    span.className = 'task-text editable';
    span.dataset.id = id;
    span.dataset.completed = task.completed;
    taskListFiltered.forEach(t => {
        if (t.id === id) span.classList.toggle('done', t.completed);
    });

    // Replace original text with span
    const originalText = taskList.querySelector(`.task-text[data-id="${id}"]`);
    const item = originalText.closest('.task-item');
    item.innerHTML = '';
    item.appendChild(span);

    // Make editable
    span.contentEditable = true;
    span.focus();

    // Save on Enter, cancel on Escape
    span.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            saveEdit(id, span.textContent.trim());
        } else if (e.key === 'Escape') {
            e.preventDefault();
            cancelEdit(id);
        }
    });

    // Click outside to save
    document.addEventListener('click', (e) => {
        if (e.target !== span && e.target.tagName !== 'SPAN') {
            saveEdit(id, span.textContent.trim());
            document.removeEventListener('click', window.handleOutsideClick);
            window.handleOutsideClick = null;
        }
    });

    // Store reference for outside click
    window.handleOutsideClick = () => {
        saveEdit(id, span.textContent.trim());
        document.removeEventListener('click', window.handleOutsideClick);
    };
}

// Save edited task
function saveEdit(id, newText) {
    if (newText.trim()) {
        const tasks = getTasks();
        const task = tasks.find(t => t.id === id);
        if (task) {
            task.text = newText;
            saveTasks(tasks);
            render();
        }
    } else {
        cancelEdit(id);
    }
}

// Cancel edit (revert or delete empty)
function cancelEdit(id) {
    const task = getTaskById(id);
    if (task) {
        const tasks = getTasks();
        taskListFiltered = filterTasks(tasks);
    }
    render();
}

// Get task by ID
function getTaskById(id) {
    const tasks = getTasks();
    return tasks.find(t => t.id === id);
}

// Get/create tasks from localStorage
function getTasks() {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
}

// Save tasks to localStorage
function saveTasks(tasks) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(tasks));
}

// Load tasks on startup
function loadTasks() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
        taskListFiltered = JSON.parse(stored);
    }
}

// Filter tasks by type
function applyFilter(filterType) {
    let tasks = getTasks();
    switch (filterType) {
        case 'all':
            taskListFiltered = tasks;
            break;
        case 'active':
            taskListFiltered = tasks.filter(t => !t.completed);
            break;
        case 'completed':
            taskListFiltered = tasks.filter(t => t.completed);
            break;
    }
    updateActiveFilter(filterType);
    render();
}

// Update active filter button
function updateActiveFilter(filterType) {
    filterBtns.forEach(btn => {
        if (btn.dataset.filter === filterType) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

// Toggle task completion
function toggleTask(id) {
    const tasks = getTasks();
    const task = tasks.find(t => t.id === id);
    if (task) {
        task.completed = !task.completed;
        saveTasks(tasks);
        render();
    }
}

// Delete a task
function deleteTask(id) {
    const tasks = getTasks();
    tasks = tasks.filter(t => t.id !== id);
    saveTasks(tasks);
    render();
}

// Clear all completed tasks
function clearCompletedTasks() {
    const tasks = getTasks();
    tasks = tasks.filter(t => !t.completed);
    saveTasks(tasks);
    render();
}

// Count remaining active tasks
function getRemainingCount() {
    const tasks = getTasks();
    return tasks.filter(t => !t.completed).length;
}

// Toggle visibility of Clear Completed button
function toggleClearButton() {
    const completedCount = getTasks().filter(t => t.completed).length;
    const shouldShow = completedCount > 0 && (filterBtns[2].dataset.filter === 'all' || 
                                              filterBtns[1].dataset.filter === 'active' ||
                                              filterBtns[2].dataset.filter === 'completed');
    clearCompletedBtn.classList.toggle('hidden', !shouldShow);
}

// Render the task list
function render() {
    updateRemainingCount();
    toggleClearButton();

    if (taskListFiltered.length === 0) {
        taskList.innerHTML = '<li class="no-tasks">' + 
            (getTasks().length === 0 ? 'No tasks yet. Add one above!' : 'No tasks found.') + 
            '</li>';
        return;
    }

    taskList.innerHTML = taskListFiltered.map(task => {
        const itemClass = task.completed ? 'task-item completed' : 'task-item';
        return `
            <li class="${itemClass}">
                <input 
                    type="checkbox" 
                    class="task-checkbox" 
                    ${task.completed ? 'checked' : ''} 
                    data-id="${task.id}"
                    onchange="toggleTask(${task.id})">
                <span 
                    class="task-text" 
                    ${task.completed ? 'done' : ''} 
                    data-id="${task.id}">
                    ${escapeHtml(task.text)}
                </span>
                <button 
                    class="task-delete" 
                    data-id="${task.id}"
                    onclick="deleteTask(${task.id})">
                    Delete
                </button>
            </li>
        `;
    }).join('');
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Update the remaining count display
function updateRemainingCount() {
    remainingCount.textContent = getRemainingCount();
}

// Start the app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
