// frontend/script.js

const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const loader = document.getElementById("loader");
const origImg = document.getElementById("origImg");
const resultImg = document.getElementById("resultImg");
const errorMsg = document.getElementById("errorMsg");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorMsg.classList.add("hidden");
  resultImg.src = "";
  
  const file = fileInput.files[0];
  if (!file) return;

  // แสดง preview ต้นฉบับ
  origImg.src = URL.createObjectURL(file);

  // เตรียมส่ง FormData
  const formData = new FormData();
  formData.append("file", file);

  loader.classList.remove("hidden");

  try {
    const resp = await fetch("/remove-outfit", {
      method: "POST",
      body: formData,
    });

    // frontend/script.js (เฉพาะบล็อกตรวจสอบ resp.ok)
    if (!resp.ok) {
     console.log("HTTP", resp.status, resp.headers.get("content-type"));

    // พยายาม parse JSON ก่อน ถ้า fail ให้ fallback อ่าน text
    let errorText;
    try {
        const errJson = await resp.json();
        errorText = errJson.detail || JSON.stringify(errJson);
    } catch {
        // ถ้าไม่ใช่ JSON หรือ body ว่าง จะเข้ามาที่นี่
        errorText = await resp.text() || resp.statusText;
    }

    throw new Error(errorText);
    }

    

    // รับเป็น Blob แล้วสร้าง Object URL ให้ img
    const blob = await resp.blob();
    resultImg.src = URL.createObjectURL(blob);

  } catch (err) {
    errorMsg.textContent = "Error: " + err.message;
    errorMsg.classList.remove("hidden");
  } finally {
    loader.classList.add("hidden");
  }
});
