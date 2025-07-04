const API_URL = "http://localhost:8000/process-image";

const fileInput  = document.getElementById("fileInput");
const shirtSelect = document.getElementById("shirtSelect");
const submitBtn  = document.getElementById("submitBtn");
const loadingDiv = document.getElementById("loading");
const bar        = document.getElementById("progressBar");
const percentTxt = document.getElementById("progressPercent");
const apiImg     = document.getElementById("apiImg");
const shirtImg   = document.getElementById("shirtImg");

shirtImg.src = `shirts/${shirtSelect.value}`;
shirtSelect.addEventListener("change", () => {
  shirtImg.src = `shirts/${shirtSelect.value}`;
});

/* ---------- Submit ---------- */
submitBtn.addEventListener("click", async () => {
  if (!fileInput.files.length) return alert("กรุณาเลือกไฟล์ภาพก่อน");

  submitBtn.disabled = true;
  loadingDiv.classList.remove("hidden");
  bar.value = 0;
  percentTxt.textContent = "0%";

  try {
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const res = await fetch(API_URL, { method: "POST", body: formData });
    if (!res.ok) throw new Error("API error");

    /* ---------- อ่านสตรีมทีละ chunk เพื่อคำนวณเปอร์เซ็นต์ ---------- */
    const contentLength = +res.headers.get("Content-Length") || 0;
    const reader = res.body.getReader();
    let received = 0;
    const chunks = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;

      // อัปเดต progress (ถ้ามี Content-Length)
      if (contentLength) {
        const pct = (received / contentLength) * 100;
        bar.value = pct;
        percentTxt.textContent = `${pct.toFixed(0)}%`;
      } else {
        // fallback – ถ้าไม่รู้ขนาด ให้วนเพิ่มช้า ๆ
        const pct = (bar.value + Math.random() * 5) % 90;
        bar.value = pct;
        percentTxt.textContent = `${pct.toFixed(0)}%`;
      }
    }

    /* ---------- รวม chunk เป็น Blob ---------- */
    const blob = new Blob(chunks, { type: "image/png" });
    apiImg.src = URL.createObjectURL(blob);

    // รีเซ็ตตำแหน่ง/ขนาด
    apiImg.style.transform = "translate(0px, 0px) scale(1)";
    apiImg.setAttribute("data-x", 500);
    apiImg.setAttribute("data-y", 500);
    apiImg.setAttribute("data-scale", 1);
    apiImg.style.width = "400px";

    // เติมเป็น 100% ทันทีเมื่อเสร็จ
    bar.value = 100;
    percentTxt.textContent = "100%";
  } catch (err) {
    alert(err.message);
  } finally {
    // ซ่อนหลังหน่วงนิดให้ผู้ใช้เห็น 100%
    setTimeout(() => {
      loadingDiv.classList.add("hidden");
      submitBtn.disabled = false;
    }, 500);
  }
});

/* ---------- Drag / Zoom ด้วย interact.js ---------- */
interact("#apiImg").draggable({
  listeners: {
    move (event) {
      const target = event.target;
      const x = (parseFloat(target.getAttribute("data-x")) || 0) + event.dx;
      const y = (parseFloat(target.getAttribute("data-y")) || 0) + event.dy;

      target.style.transform = `translate(${x}px, ${y}px) scale(${target.getAttribute("data-scale") || 1})`;
      target.setAttribute("data-x", x);
      target.setAttribute("data-y", y);
    }
  }
}).gesturable({
  listeners: {
    move (event) {
      const target = event.target;
      const currentScale = parseFloat(target.getAttribute("data-scale")) || 1;
      const newScale = currentScale * (1 + event.ds);

      target.style.transform = `translate(${target.getAttribute("data-x") || 0}px, ${target.getAttribute("data-y") || 0}px) scale(${newScale})`;
      target.setAttribute("data-scale", newScale);
    }
  }
}).resizable({
    // ให้ลากจากขอบขวา-ล่าง (preserveAspectRatio จะคงสัดส่วนให้)
    edges: { left: false, right: true, bottom: true, top: false },
    preserveAspectRatio: true
  })
  .on('resizemove', event => {
      const target = event.target;
      // ขนาดใหม่ที่ interact.js คำนวณให้
      const { width, height } = event.rect;
      target.style.width  = width  + 'px';
      target.style.height = height + 'px';
  });
