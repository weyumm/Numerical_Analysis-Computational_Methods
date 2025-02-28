class Particle {
    constructor(x, y, char) {
        this.x = x;
        this.y = y;
        this.char = char;
        this.baseX = x;
        this.baseY = y;
        this.density = (Math.random() * 30) + 1;
        this.size = 16;
        this.distance;
        this.angle = Math.random() * Math.PI * 2;
        this.velocity = 0;
        this.heartPosition = Math.random() * Math.PI * 2;
    }

    draw(ctx) {
        const distToMouse = Math.sqrt(
            Math.pow(mouse.x - this.x, 2) + 
            Math.pow(mouse.y - this.y, 2)
        );
        
        if (distToMouse < 30) {
            ctx.fillStyle = 'rgba(255, 20, 147, 1)';
        } else if (distToMouse < 60) {
            ctx.fillStyle = 'rgba(255, 105, 180, 0.9)';
        } else {
            ctx.fillStyle = 'black';
        }
        
        ctx.font = '20px "Times New Roman"';
        ctx.fillText(this.char, this.x, this.y);
    }

    update(mouse) {
        let dx = mouse.x - this.x;
        let dy = mouse.y - this.y;
        let distance = Math.sqrt(dx * dx + dy * dy);
        
        const minDistance = 30;
        const gravitationalPull = isHeartActive ? 4 : 2;
        
        if (distance < maxGravityDistance) {
            let force = (1 - distance / maxGravityDistance) * gravitationalPull;
            
            if (distance > minDistance) {
                let forceDirectionX = dx / distance;
                let forceDirectionY = dy / distance;
                
                this.angle += 0.05 * force;
                
                this.x += forceDirectionX * force * this.density;
                this.y += forceDirectionY * force * this.density;
                
                this.x += Math.cos(this.angle) * force;
                this.y += Math.sin(this.angle) * force;
            } else {
                const t = this.heartPosition;
                // 使用动态计算的心形大小
                const scale = isHeartActive ? heartScale : 15;
                
                const targetX = mouse.x + scale * (16 * Math.pow(Math.sin(t), 3));
                const targetY = mouse.y - scale * (13 * Math.cos(t) - 5 * Math.cos(2*t) - 2 * Math.cos(3*t) - Math.cos(4*t));
                
                const dxHeart = targetX - this.x;
                const dyHeart = targetY - this.y;
                const distHeart = Math.sqrt(dxHeart * dxHeart + dyHeart * dyHeart);
                
                if (distHeart > 0.1) {
                    this.x += dxHeart * 0.1;
                    this.y += dyHeart * 0.1;
                } else {
                    this.x = targetX;
                    this.y = targetY;
                }
            }
        } else {
            if (this.x !== this.baseX) {
                let dx = this.x - this.baseX;
                this.x -= dx/20;
            }
            if (this.y !== this.baseY) {
                let dy = this.y - this.baseY;
                this.y -= dy/20;
            }
        }
    }
}

// 默认文本
const defaultText = `你知道吗，我们见到的太阳是八分钟之前的太阳，见到的月亮是一点三秒之前的月亮，见到一英里之外的建筑是五微秒之前存在的，即使你在我一米之外，我见到的也是三纳米秒以前的你。我们所眼见的都是过去，而一切也都会过去。

我大约真的没有什么才华，
只是因为有幸见着了你，
于是这颗庸常的心中才凭空生出好些浪漫。

从我拿起笔，
准备叙述你的细节开始，
总是忍不住走神，
真抱歉，情话没写出来，
可我实实在在地想了你一个小时。`;

// 使用本地存储保存上次编辑的内容
let text = localStorage.getItem('heartText') || defaultText;

let particles = [];
let mouse = {
    x: null,
    y: null,
    radius: 100
};

let squareSize = 800;
let isHeartActive = false;
let maxGravityDistance = 200;
// 新增心形大小动态调整变量
let heartScale = 25;

// 声明编辑功能相关变量
let textContent;
let hint;
let isEditing = false;

function drawBlackHoleEffect(ctx, x, y) {
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, 200);
    gradient.addColorStop(0, 'rgba(255, 20, 147, 0.3)');
    gradient.addColorStop(0.5, 'rgba(255, 105, 180, 0.1)');
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
    
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, 200, 0, Math.PI * 2);
    ctx.fill();
}

function init() {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    particles = [];
    
    const totalCharsCount = text.replace(/\s/g, '').length;
    const isLongText = totalCharsCount > 800;
    
    let displayWidth, displayHeight, displayX, displayY;
    let charWidth = 18;
    
    if (isLongText) {
        const margin = Math.min(window.innerWidth, window.innerHeight) * 0.05;
        displayWidth = window.innerWidth - 2 * margin;
        displayHeight = window.innerHeight - 2 * margin;
        displayX = margin;
        displayY = margin;
        
        charWidth = totalCharsCount > 1500 ? 16 : (totalCharsCount > 1000 ? 17 : 18);
    } else {
        squareSize = Math.min(800, Math.min(window.innerWidth, window.innerHeight) * 0.8);
        displayWidth = squareSize;
        displayHeight = squareSize;
        displayX = canvas.width/2 - squareSize/2;
        displayY = canvas.height/2 - squareSize/2;
    }
    
    ctx.textAlign = 'left';
    ctx.font = '20px "Times New Roman"';
    
    const lineHeight = 40;
    const maxCharsPerLine = Math.floor((displayWidth - 60) / charWidth);
    let formattedLines = [];
    const inputText = text.trim();
    const paragraphs = inputText.split('\n');
    
    paragraphs.forEach(paragraph => {
        if (paragraph.length <= maxCharsPerLine) {
            formattedLines.push(paragraph);
        } else {
            let currentLine = '';
            for (let i = 0; i < paragraph.length; i++) {
                const char = paragraph[i];
                
                if (currentLine.length >= maxCharsPerLine) {
                    formattedLines.push(currentLine);
                    currentLine = char;
                } else {
                    currentLine += char;
                }
            }
            if (currentLine.length > 0) {
                formattedLines.push(currentLine);
            }
        }
    });

    const maxLines = Math.floor((displayHeight - 80) / lineHeight);
    if (formattedLines.length > maxLines) {
        formattedLines = formattedLines.slice(0, maxLines);
    }

    let nonSpaceChars = [];
    formattedLines.forEach(line => {
        line.split('').forEach(char => {
            if (char.trim() !== '') {
                nonSpaceChars.push(char);
            }
        });
    });
    
    const totalChars = nonSpaceChars.length;
    
    if (totalChars > 1800) {
        heartScale = 65;
    } else if (totalChars > 1200) {
        heartScale = 60;
    } else if (totalChars > 1000) {
        heartScale = 55; 
    } else if (totalChars > 800) {
        heartScale = 50;
    } else if (totalChars > 600) {
        heartScale = 45;
    } else if (totalChars > 400) {
        heartScale = 40;
    } else if (totalChars > 200) {
        heartScale = 35;
    } else if (totalChars > 100) {
        heartScale = 30;
    } else {
        heartScale = 25;
    }
    
    const maxHeartChars = 1200; 
    const useSampling = totalChars > maxHeartChars;
    const sampleRate = useSampling ? totalChars / maxHeartChars : 1;
    
    let heartPositionCounter = 0;
    let sampledCharCount = 0;
    
    formattedLines.forEach((line, lineIndex) => {
        const characters = line.split('');
        
        let lineX;
        if (isLongText) {
            lineX = displayX + 30; 
        } else {
            const lineWidth = Math.min(characters.length * charWidth, displayWidth - 60);
            lineX = displayX + (displayWidth - lineWidth) / 2;
        }

        characters.forEach((char, i) => {
            const x = lineX + (i * charWidth);
            const y = displayY + (lineIndex * lineHeight) + 40;
            
            const inDisplayArea = isLongText || 
                                  (x >= displayX && x < displayX + displayWidth - 20 && 
                                   y >= displayY && y < displayY + displayHeight - 20);
                                   
            if (inDisplayArea) {
                const particle = new Particle(x, y, char);
                
                if (char.trim() !== '') {
                    if (!useSampling || (heartPositionCounter % Math.round(sampleRate)) === 0) {
                        particle.heartPosition = (sampledCharCount / Math.min(totalChars, maxHeartChars)) * Math.PI * 2;
                        sampledCharCount++;
                    } else {
                        particle.heartPosition = Math.random() * Math.PI * 2;
                    }
                    heartPositionCounter++;
                } else {
                    particle.heartPosition = Math.random() * Math.PI * 2;
                }
                
                particles.push(particle);
            }
        });
    });
}



function applyTextChange() {
    const newText = textContent.textContent.trim();
    if (newText) {
        text = newText;
        localStorage.setItem('heartText', text);
        particles = [];
        init();
        
        if (isHeartActive) {
            mouse.x = window.innerWidth / 2;
            mouse.y = window.innerHeight / 2;
            maxGravityDistance = 3000;
        }
    }
    textContent.style.display = 'none';
    isEditing = false;
}

window.addEventListener('click', () => {
    if (isEditing) return;
    
    isHeartActive = !isHeartActive;
    
    if (isHeartActive) {
        mouse.x = window.innerWidth / 2;
        mouse.y = window.innerHeight / 2;
        maxGravityDistance = 3000; 
    } else {
        mouse.x = null;
        mouse.y = null;
        maxGravityDistance = 200;
    }
});

if (window.mousemoveHandler) {
    window.removeEventListener('mousemove', window.mousemoveHandler);
}

window.mousemoveHandler = (event) => {
    if (!isHeartActive) {
        mouse.x = event.x;
        mouse.y = event.y;
    }
};

window.addEventListener('mousemove', window.mousemoveHandler);

function animate() {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const isLongText = particles.length > 0 && 
                       particles.filter(p => p.char.trim() !== '').length > 800;
    
    let displayX, displayY, displayWidth, displayHeight;
    if (isLongText) {
        const margin = Math.min(window.innerWidth, window.innerHeight) * 0.05;
        displayWidth = window.innerWidth - 2 * margin;
        displayHeight = window.innerHeight - 2 * margin;
        displayX = margin;
        displayY = margin;
    } else {
        displayWidth = squareSize;
        displayHeight = squareSize;
        displayX = canvas.width/2 - squareSize/2;
        displayY = canvas.height/2 - squareSize/2;
    }
    
    if (mouse.x !== null && mouse.y !== null) {
        drawBlackHoleEffect(ctx, mouse.x, mouse.y);
    }

    if (!isLongText) {
        ctx.strokeStyle = 'rgba(255, 105, 180, 0)';
        ctx.lineWidth = 2;
        ctx.strokeRect(displayX, displayY, displayWidth, displayHeight);
    }

    particles.forEach(particle => {
        if (isLongText) {
            const padding = 40;
            particle.x = Math.max(-padding, Math.min(canvas.width + padding, particle.x));
            particle.y = Math.max(-padding, Math.min(canvas.height + padding, particle.y));
        } else {
            particle.x = Math.max(displayX, Math.min(displayX + displayWidth, particle.x));
            particle.y = Math.max(displayY, Math.min(displayY + displayHeight, particle.y));
        }
        
        particle.update(mouse);
        particle.draw(ctx);
    });
    
    requestAnimationFrame(animate);
}

window.addEventListener('resize', () => {
    particles = [];
    init();
    
    if (isHeartActive) {
        mouse.x = window.innerWidth / 2;
        mouse.y = window.innerHeight / 2;
    }
});

window.onload = function() {
    textContent = document.getElementById('text-content');
    hint = document.getElementById('hint');
    
    textContent.textContent = text;
    
    window.addEventListener('dblclick', function() {
        if (!isEditing) {
            isEditing = true;
            textContent.style.display = 'block';
            textContent.focus();
            
            const range = document.createRange();
            const selection = window.getSelection();
            range.selectNodeContents(textContent);
            range.collapse(false); 
            selection.removeAllRanges();
            selection.addRange(range);
        }
    });
    
    textContent.addEventListener('blur', applyTextChange);
    
    textContent.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            applyTextChange();
        } else if (e.key === 'Escape') {
            textContent.textContent = text;
            textContent.style.display = 'none';
            isEditing = false;
        }
    });
    
    setTimeout(function() {
        hint.classList.add('fade');
    }, 3000);
    
    init();
    animate();
};