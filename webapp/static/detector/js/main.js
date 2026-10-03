// 前端交互: 图片预览 / 拖拽上传 / 模型卡片选中 / 置信度滑块 / 提交遮罩 / 历史删除
document.addEventListener("DOMContentLoaded", function () {

    /* 1. 图片实时预览 + 文件名标签 */
    var input = document.getElementById("image-input");
    var preview = document.getElementById("preview");
    var empty = document.getElementById("preview-empty");
    var pill = document.getElementById("file-pill");
    var pillText = document.getElementById("file-name-text");

    function showFile(file) {
        if (!file) return;
        if (!file.type.startsWith("image/")) {
            alert("请选择图片文件");
            return;
        }
        if (pillText && pill) {
            pillText.textContent = file.name + "（" + (file.size / 1048576).toFixed(2) + " MB）";
            pill.classList.add("show");
        }
        var reader = new FileReader();
        reader.onload = function (e) {
            if (preview) {
                preview.src = e.target.result;
                preview.classList.remove("d-none");
            }
            if (empty) empty.classList.add("d-none");
        };
        reader.readAsDataURL(file);
    }

    if (input) {
        input.addEventListener("change", function () {
            var file = this.files && this.files[0];
            if (file) showFile(file);
        });
    }

    /* 2. 拖拽上传 */
    var zone = document.getElementById("upload-zone");
    if (zone && input) {
        var dragDepth = 0;
        zone.addEventListener("dragenter", function (e) {
            e.preventDefault();
            dragDepth++;
            zone.classList.add("dragover");
        });
        zone.addEventListener("dragover", function (e) { e.preventDefault(); });
        zone.addEventListener("dragleave", function (e) {
            e.preventDefault();
            dragDepth = Math.max(0, dragDepth - 1);
            if (dragDepth === 0) zone.classList.remove("dragover");
        });
        zone.addEventListener("drop", function (e) {
            e.preventDefault();
            dragDepth = 0;
            zone.classList.remove("dragover");
            var file = e.dataTransfer.files && e.dataTransfer.files[0];
            if (!file) return;
            if (!file.type.startsWith("image/")) {
                alert("请拖拽图片文件");
                return;
            }
            var dt = new DataTransfer();
            dt.items.add(file);
            input.files = dt.files;
            showFile(file);
        });
    }

    /* 3. 模型卡片选中态联动（四模型） */
    var SELECTED_CLASS = {
        yolov8: "selected-v8", yolov8n: "selected-v8",
        yolov11: "selected-v11", yolov11n: "selected-v11",
        both: "selected-both", nano: "selected-both", all: "selected-both"
    };
    function syncModelOptions() {
        document.querySelectorAll(".model-option").forEach(function (label) {
            label.classList.remove("selected-v8", "selected-v11", "selected-both");
            var radio = label.querySelector("input[type=radio]");
            if (radio && radio.checked && label.dataset.key) {
                var cls = SELECTED_CLASS[label.dataset.key];
                if (cls) label.classList.add(cls);
            }
        });
    }
    document.querySelectorAll('input[name="models"]').forEach(function (radio) {
        radio.addEventListener("change", syncModelOptions);
    });
    syncModelOptions();

    /* 4. 置信度滑块联动 */
    var slider = document.getElementById("conf");
    var label = document.getElementById("conf-value");
    if (slider && label) {
        slider.addEventListener("input", function () {
            label.textContent = parseFloat(this.value).toFixed(2);
        });
    }

    /* 5. 提交时显示加载遮罩 */
    var form = document.getElementById("detect-form");
    var loading = document.getElementById("loading");
    if (form && loading) {
        form.addEventListener("submit", function () {
            if (input && input.files && input.files.length > 0) {
                loading.classList.remove("d-none");
            }
        });
    }

    /* 6. 历史记录删除（带确认 + CSRF 令牌） */
    function getCookie(name) {
        var m = document.cookie.match("(^|;)\\s*" + name + "=([^;]*)");
        return m ? m.pop() : "";
    }
    document.querySelectorAll(".btn-delete").forEach(function (btn) {
        btn.addEventListener("click", function () {
            if (!confirm("确定删除该条检测记录吗？")) return;
            fetch(btn.dataset.url, {
                method: "POST",
                headers: {
                    "X-Requested-With": "XMLHttpRequest",
                    "X-CSRFToken": getCookie("csrftoken")
                }
            }).then(function () { location.reload(); });
        });
    });
});
