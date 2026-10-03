# Todo List App

A simple, dependency-free todo list web application.

## How to Open

Open `index.html` in any web browser:

1. **Double-click** the `index.html` file, or
2. Open your browser and go to `file:///full/path/to/index.html`, or
3. In a terminal, run:
   ```bash
   python3 -m http.server 8000
   ```
   Then open `http://localhost:8000` in your browser.

## Features

- **Add tasks** with the Enter key or the Add button (empty text is ignored)
- **Mark tasks done** with checkboxes (completed tasks appear crossed out)
- **Edit tasks** by double-clicking (Enter saves, Escape cancels)
- **Delete tasks** with the Delete button
- **Filter tasks**: All, Active, Completed
- **Task counter**: Shows how many tasks remain active
- **Clear completed**: Remove all completed tasks at once
- **LocalStorage**: Your tasks persist across browser reloads

## Files

- `index.html` - Main HTML file
- `style.css` - All styles (no frameworks)
- `app.js` - All JavaScript logic (no frameworks, small functions)
- `README.md` - This file
