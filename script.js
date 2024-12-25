// 获取必要的 DOM 元素
const floatingWindow = document.getElementById("floatingWindow");
const header = floatingWindow.querySelector(".window-header");
const minimizeButton = document.getElementById("minimizeButton");
const runButton = document.getElementById("runButton");
const openWindowButton = document.getElementById("openWindowButton");
const consoleContainer = document.getElementById("console-container");
const editorContainer = document.getElementById("editor-container");
let isDragging = false;
let offsetX, offsetY;

// 初始化 CodeMirror
const editor = CodeMirror(editorContainer, {
    mode: "python",
    theme: "dracula",
    lineNumbers: true,
    matchBrackets: true,
    styleActiveLine: true,
});

// 默认隐藏悬浮窗口
floatingWindow.classList.add("minimized");
openWindowButton.classList.remove("hidden");

// 打开窗口按钮功能
openWindowButton.addEventListener("click", () => {
    floatingWindow.classList.remove("minimized");
    openWindowButton.classList.add("hidden");
});

// 设置编辑器大小为父容器大小
editor.setSize("100%", "100%");

// 开始拖动
header.addEventListener("mousedown", (e) => {
    isDragging = true;
    offsetX = e.clientX - floatingWindow.offsetLeft;
    offsetY = e.clientY - floatingWindow.offsetTop;
    header.style.cursor = "grabbing";
});

// 拖动中
document.addEventListener("mousemove", (e) => {
    if (!isDragging) return;

    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    let x = e.clientX - offsetX;
    let y = e.clientY - offsetY;

    const maxX = viewportWidth - floatingWindow.offsetWidth;
    const maxY = viewportHeight - floatingWindow.offsetHeight;

    x = Math.max(0, Math.min(x, maxX));
    y = Math.max(0, Math.min(y, maxY));

    floatingWindow.style.left = `${x}px`;
    floatingWindow.style.top = `${y}px`;
});

// 停止拖动
document.addEventListener("mouseup", () => {
    isDragging = false;
    header.style.cursor = "grab";
});

// 最小化按钮功能
minimizeButton.addEventListener("click", () => {
    floatingWindow.classList.add("minimized");
    openWindowButton.classList.remove("hidden");
});

// 运行按钮功能
runButton.addEventListener("click", async () => {
    if (!pyodide) {
        await initializePyodide();
    }
    const code = editor.getValue();
    consoleContainer.innerText = "执行中...";
    try {
        // 动态加载必要的包
        await pyodide.loadPackage(["scikit-learn", "pandas", "numpy"]);

        // 重定向输出
        pyodide.runPython(`
import sys
from io import StringIO
sys.stdout = StringIO()
sys.stderr = StringIO()
`);
        // 执行用户代码
        pyodide.runPython(code);
        const output = pyodide.runPython("sys.stdout.getvalue()");
        consoleContainer.innerText = output || "无输出";
    } catch (error) {
        consoleContainer.innerText = `错误: ${error.message}`;
    }
});


// 初始化 Pyodide 和加载库
let pyodide;
async function initializePyodide() {
    if (!pyodide) {
        consoleContainer.innerText = "正在加载 Pyodide 和依赖库...";

        // 加载 Pyodide 主脚本
        pyodide = await loadPyodide();

        // 动态安装 micropip 和其他包
        await pyodide.loadPackage(["numpy", "pandas", "matplotlib", "seaborn"]);
        await pyodide.loadPackage("micropip");
        await pyodide.runPythonAsync(`
            import micropip
            await micropip.install("jieba")
        `);

        consoleContainer.innerText = "Pyodide 和库加载完成！";
    }
}

// 加载 Pyodide 主函数
async function loadPyodide() {
    if (!window.pyodide) {
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/pyodide/v0.23.4/full/pyodide.js";
        document.body.appendChild(script);
        await new Promise((resolve, reject) => {
            script.onload = resolve;
            script.onerror = reject;
        });
        window.pyodide = await window.loadPyodide();
    }
    return window.pyodide;
}

// 初始化时静态加载库
initializePyodide();
