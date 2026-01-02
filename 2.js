/**
 * 运行方式: node extract.js "你的豆瓣搜索URL"
 */
const https = require('https');

// 从命令行参数获取 URL
const url = process.argv[2];

if (!url) {
    console.log('使用说明: node extract.js "https://www.douban.com/search?cat=1002&q=小姨"');
    process.exit(1);
}

/**
 * 纯正则解析逻辑 (处理纯文本字符串)
 * @param {string} htmlRaw 
 */
function parseDoubanData(htmlRaw) {
    // 1. 将 HTML 文本按电影条目块分割
    const blocks = htmlRaw.split('<div class="result">').slice(1);

    const movies = blocks.map(block => {
        // --- 提取 ID (sid: 数字) ---
        const idMatch = block.match(/sid:\s*(\d+)/);
        let id = idMatch ? idMatch[1] : null;

        const discardMatch = block.match(/\((尚未播出|尚未上映)\)/);
        if (discardMatch) id = null;

        id = parseInt(id);

        // --- 提取海报链接 ---
        const posterMatch = block.match(/<img src="(https:\/\/img[^"]+\.jpg)"/);
        const img = posterMatch ? posterMatch[1] : "";

        // --- 提取影片名 ---
        const titleMatch = block.match(/onclick="[^"]+" >([^<]+)<\/a>/);
        let name = titleMatch ? titleMatch[1].trim() : "";

        // --- 提取评分 (匹配 class="rating_nums" 里的数字) ---
        // 如果找不到数字，则尝试匹配“暂无评分”或“尚未播出”等描述
        let rate = "0";
        const ratingMatch = block.match(/class="rating_nums">([\d.]+)<\/span>/);
        if (ratingMatch) {
            rate = ratingMatch[1];
        } else {
            const noRatingMatch = block.match(/\((暂无评分|尚未播出|尚未上映)\)/);
            if (noRatingMatch) rate = '0.0';
        }

        // --- 提取包含原名和年份的混合文本行 ---
        const castMatch = block.match(/<span class="subject-cast">([^<]+)<\/span>/);
        const castLine = castMatch ? castMatch[1].trim() : "";

        // --- 解析年份 (文本行最后的 4 位数字) ---
        const yearMatch = castLine.match(/(\d{4})$/);
        const year = yearMatch ? yearMatch[1] : "未知";

        // --- 解析原名 ---
        let originalName = name;
        if (castLine.includes("原名:")) {
            const origMatch = castLine.match(/原名:([^/]+)/);
            if (origMatch) {
                originalName = origMatch[1].trim();
            }
        }

        return {
            id,
            name,
            rate,
            img,
            originalName,
            year
        };
    }).filter(m => m.id);

    return movies;
}

// 请求配置
const options = {
    headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
};

// 发起网络请求获取纯文本源码
https.get(url, options, (res) => {
    res.setEncoding('utf8'); // 确保以纯文本字符串处理
    let rawHtml = '';

    res.on('data', (chunk) => { rawHtml += chunk; });

    res.on('end', () => {
        if (res.statusCode !== 200) {
            console.error(`请求失败，状态码: ${res.statusCode}`);
            return;
        }

        const resultJson = parseDoubanData(rawHtml);
        console.log(JSON.stringify(resultJson, null, 2));
    });

}).on('error', (e) => {
    console.error(`网络请求出错: ${e.message}`);
});