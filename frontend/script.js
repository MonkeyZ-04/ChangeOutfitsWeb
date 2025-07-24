document.addEventListener('DOMContentLoaded', () => {
    // --- (สำคัญ) แก้ไขลิสต์เสื้อของคุณตรงนี้ ---
    const shirts = [
        { name: "เสื้อ Security สีเทา", file: "guard_gray.png" },
        { name: "เสื้อ Security สีเทา มี เนคไท", file: "guard_tie.png" },
        { name: "เสื้อยืดสีขาว ผช", file: "m_white.png" },
        { name: "เสื้อ Security ผู้หญิง", file: "guard_women.png" },
        { name: "เสื้อ Security สีทอง แซมโก้", file: "guard_gold_samco.png" },
        { name: "เสื้อยืดสีขาว ผญ", file: "f_white.png" },
        // เพิ่มเสื้อตัวอื่นๆ ตามฟอร์แมตนี้
        // ตัวอย่าง:
        // { name: "เสื้อเชิ้ตสีฟ้า", file: "shirt-blue.jpg" },
        // { name: "เสื้อโปโลสีดำ", file: "polo-black.png" },
    ];
    // -----------------------------------------

    // อ้างอิงถึง Element ทั้งหมดที่ต้องใช้
    const uploadInput = document.getElementById('image-upload');
    const uploadBtn = document.getElementById('upload-btn');
    const loader = document.getElementById('loader');
    const resultContainer = document.getElementById('result-container');
    const personImg = document.getElementById('person-img');
    const shirtImg = document.getElementById('shirt-img');
    const shirtSelector = document.getElementById('shirt-selector');
    const controlsSection = document.getElementById('controls-section');
    const scaleSlider = document.getElementById('scale-slider');
    const xSlider = document.getElementById('x-slider');
    const ySlider = document.getElementById('y-slider');
    const resetBtn = document.getElementById('reset-btn');
    const downloadBtn = document.getElementById('download-btn');

    const API_URL = 'http://127.0.0.1:8000/process-image';

    // สร้างตัวเลือกใน Dropdown จากลิสต์ shirts
    shirts.forEach(shirt => {
        const option = document.createElement('option');
        option.value = shirt.file;
        option.textContent = shirt.name;
        shirtSelector.appendChild(option);
    });
    // ตั้งค่าเสื้อตัวแรกเป็นค่าเริ่มต้น
    if (shirts.length > 0) {
        shirtImg.src = `images/${shirts[0].file}`;
    }

    // Event Listener เมื่อมีการเลือกเสื้อจาก Dropdown
    shirtSelector.addEventListener('change', (event) => {
        const selectedShirtFile = event.target.value;
        shirtImg.src = `images/${selectedShirtFile}`;
    });

    // ซ่อนส่วนควบคุมและผลลัพธ์ไว้ก่อน
    controlsSection.style.display = 'none';
    resultContainer.style.display = 'none';

    function updateImageTransform() {
        const scaleValue = scaleSlider.value;
        const xValue = xSlider.value;
        const yValue = ySlider.value;
        personImg.style.transform = `translateX(${xValue}px) translateY(${yValue}px) scale(${scaleValue})`;
    }

    function downloadImage() {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = shirtImg.naturalWidth;
        canvas.height = shirtImg.naturalHeight;
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(shirtImg, 0, 0, canvas.width, canvas.height);

        const scaleValue = parseFloat(scaleSlider.value);
        const scaleFactor = shirtImg.naturalWidth / shirtImg.clientWidth;
        const xValue = parseInt(xSlider.value) * scaleFactor;
        const yValue = parseInt(ySlider.value) * scaleFactor;
        ctx.save();
        const personCenterX = (canvas.width / 2) + xValue;
        const personCenterY = (canvas.height / 2) + yValue;
        ctx.translate(personCenterX, personCenterY);
        ctx.scale(scaleValue, scaleValue);
        ctx.translate(-canvas.width / 2, -canvas.height / 2);
        ctx.drawImage(personImg, 0, 0, canvas.width, canvas.height);
        ctx.restore();
        ctx.drawImage(shirtImg, 0, 0, canvas.width, canvas.height);
        const link = document.createElement('a');
        link.download = 'new-outfit.png';
        link.href = canvas.toDataURL('image/png');
        link.click();
    }

    // --- เพิ่ม Event Listeners ---
    scaleSlider.addEventListener('input', updateImageTransform);
    xSlider.addEventListener('input', updateImageTransform);
    ySlider.addEventListener('input', updateImageTransform);
    resetBtn.addEventListener('click', () => {
        scaleSlider.value = 1;
        xSlider.value = 0;
        ySlider.value = 0;
        updateImageTransform();
    });
    downloadBtn.addEventListener('click', downloadImage);

    // Event Listener หลักสำหรับปุ่ม "ประมวลผลรูปภาพ"
    uploadBtn.addEventListener('click', async () => {
        const file = uploadInput.files[0];
        if (!file) {
            alert('กรุณาเลือกไฟล์รูปภาพก่อน');
            return;
        }
        loader.style.display = 'block';
        resultContainer.style.display = 'none';
        controlsSection.style.display = 'none';
        resetBtn.click();
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'เกิดข้อผิดพลาดในการประมวลผล');
            }
            const imageBlob = await response.blob();
            const imageUrl = URL.createObjectURL(imageBlob);
            personImg.src = imageUrl;
            resultContainer.style.display = 'block';
            controlsSection.style.display = 'flex';
        } catch (error) {
            console.error('Error:', error);
            alert(`เกิดข้อผิดพลาด: ${error.message}`);
        } finally {
            loader.style.display = 'none';
        }
    });
});