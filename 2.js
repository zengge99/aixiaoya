import { connect } from 'cloudflare:sockets';

export default {
  async fetch(request, env, ctx) {
    return handleRequest(request);
  },
};

let isInCf = true;
function cfLog(...vars) {
  if (isInCf) {
    console.log(...vars);
  }
}

async function transformHTMLtoJSON(html, filters) {
  //return html;
  let stack = [];
  let output = {};
  stack.push(output);
  class HTMLTransformer {
    element(el) {
      let curTag = el.tagName;
      let cur = stack[stack.length - 1];
      if (!cur[curTag]) {
        cur[curTag] = [];
      }
      let jsonEl = {};
      cur.fullpath = cur.fullpath || "";
      jsonEl.fullpath = cur.fullpath + "/" + el.tagName;
      [...el.attributes].map(([k, v]) => jsonEl[k] = v);
      stack.push(jsonEl);
      cur[curTag].push(jsonEl);
      try {
        el.onEndTag(endTag => {
          stack.pop();
        });
      } catch {
        stack.pop();
      }
    }
    text(text) {
      let cur = stack[stack.length - 1];
      if (!cur['text_content']) {
        cur['text_content'] = '';
      }
      cur['text_content'] += '\n' + text.text;
      cur['text_content'] = cur['text_content'].replace(/\n+/g, '\n').trim();
    }
  }
  let parser = new HTMLRewriter();
  filters.forEach(element => {
    parser = parser.on(element, new HTMLTransformer());
  });

  await parser.transform(new Response(html)).text();
  return output;
}

function formatDoubanJson(input) {
  input = input.a;
  let output = [];
  input.forEach(element => {
    if (element.href && element.href.includes("movie")) {
      try {
        if (element.text_content && (
                element.text_content.includes("尚未播出") || 
                element.text_content.includes("尚未上映")
              )
            ) {
          return;
        }
        let doubanInfo = {};
        let regex = /\d+/g;
        let match = element.href.match(regex);
        doubanInfo.id = element.href = parseInt(match[0]);
        doubanInfo.name = element.text_content.split('\n')[0];
        regex = /\d+(\.\d+)?/g
        match = element.text_content.split('\n')[1].match(regex);
        doubanInfo.rate = match && match[0] ? match[0] : '0.0';
        doubanInfo.img = element.img[0].src;
        doubanInfo.similarity = TextUtils.similarity(keyword, doubanInfo.name);
        output.push(doubanInfo);
      } catch { }
    }
  });
  output.sort((a, b) => b.similarity - a.similarity);
  output.push({ keyword: keyword });
  return output;
}

class TextUtils {

  static NUMBERS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"];
  static NUMBER = /第(\d+\.?\d*)季/;
  static NUMBER2 = /(第.+季)/;
  static NUMBER3 = / ?(S\d{1,2})/;
  static NUMBER4 = /(\.?\d+集全?)/;
  static NUMBER5 = /(\.[0-9-]+季(\+番外)?)/;
  static NUMBER6 = /(.更新至\d+)/;
  static NAME1 = /^【(.+)】$/;
  static NAME2 = /^\w (.+)\s+\(\d{4}\).*$/;
  static NAME3 = /^\w (.+)\.\d{4} .+$/;

  static isChineseChar(c) {
    return c >= 0x4E00 && c <= 0x9FA5;
  }

  static isChineseChar2(c) {
    return (c >= 0x4E00 && c <= 0x9FA5) || (c >= '0'.codePointAt(0) && c <= '9'.codePointAt(0)) || c == '：'.codePointAt(0);
  }

  static isEnglishChar(c) {
    return (c >= 'A'.codePointAt(0) && c <= 'Z'.codePointAt(0));
  }

  static isChinese(text) {
    return [...text].every(c => TextUtils.isChineseChar2(c.codePointAt(0)));
  }

  static isEnglish(text) {
    if (/^[a-zA-Z ]+$/.test(text)) {
      return true; // 如果字符串是纯英文，则直接返回
    }
    return false;
  }

  static fixName(name) {
    let newName = name.trim();
    let index = newName.lastIndexOf('/');
    if (index > -1) {
      newName = newName.substring(index + 1);
    }

    cfLog("pos1", newName);
    //去扩展名
    newName = TextUtils.removeEnglishExtension(newName);

    //特殊字符替换
    newName = newName
      .replace(/x265/g, " ")
      .replace("5.1", " ")
      .replace("10bit", " ")
      .replace(/豆瓣\d+(\.\d+)?分?/g, ' ')
      .replace("一只鱼\s?4kyu.cc", " ")
      .replace(/H\.264/g, " ")
      .replace(/Top\d+\./g, ' ')
      .replace(/\./g, "  ")
      .replace(/&/g, "  ")
      .replace(/\s美国\s?/g, ' ')
      .replace(/\s中国\s?/g, ' ')
      .replace(/\s日本\s?/g, ' ')
      .replace(/\s德国\s?/g, ' ')
      .replace(/\s法国\s?/g, ' ')
      .replace(/\s英国\s?/g, ' ')
      .replace(/\s加拿大\s?/g, ' ')
      .replace(/\s意大利\s?/g, ' ')
      .replace(/\s西班牙\s?/g, ' ')
      .replace(/\s澳大利亚\s?/g, ' ')
      .replace(/\s墨西哥\s?/g, ' ')
      .replace(/\s巴西\s?/g, ' ')
      .replace(/\s俄罗斯\s?/g, ' ')
      .replace(/\s印度\s?/g, ' ')
      .replace(/\s韩国\s?/g, ' ')
      .replace(/\s荷兰\s?/g, ' ')
      .replace(/\s瑞典\s?/g, ' ')
      .replace(/\s瑞士\s?/g, ' ')
      .replace(/\s新加坡\s?/g, ' ')
      .replace(/\s南非\s?/g, ' ')
      .replace(/\s泰国\s?/g, ' ');
    cfLog("pos7", newName);

    if (newName.endsWith(")")) {
      index = newName.lastIndexOf('(');
      if (index > 0) {
        newName = newName.substring(0, index);
      }
    }

    //去掉括号结尾的部分
    if (newName.endsWith("）")) {
      index = newName.lastIndexOf('（');
      if (index > 0) {
        newName = newName.substring(0, index);
      }
    }

    let start = newName.indexOf('《');
    if (start > -1) {
      let end = newName.indexOf('》', index + 1);
      if (end > start) {
        newName = newName.substring(start + 1, end);
      }
    }

    cfLog("pos8", newName);

    //中英文混合，提取中文部分。第一部分是中文，后面是英文
    let parts = newName.replace("[", "").replace("]", " ").replace(/\s+/g, " ").split(" ");
    if (parts.length > 4 && (parts[0].length > 1 && parts[1].length > 1 && TextUtils.isChinese(parts[0]))) {
      cfLog("pos12", newName);
      //if (TextUtils.isEnglish(parts[1])) {
      if (!TextUtils.isChinese(parts[1])) {
        newName = parts[0];
        cfLog("pos10", newName);
      }
    }

    cfLog("pos9", newName);

    newName = newName
      .replace(/1~\d{1,2}/g, " ")
      .replace(/S\d{1,2}-\d{1,2}/g, " ")
      .replace(/1-\d+[集季部]/g, " ")
      .replace(/共\d+集\+\d+部剧场版/g, " ")
      .replace(/^\d{4}/g, " ")
      .replace(/\[\d{4,8}\]/, "")
      .replace(/\d{4,8}/, "")
      .replace(/(?<!^)\[.+\]/g, " ")
      .replace(/1-\d{1,2}/g, " ")
      .replace(/\(.+\)/g, "")
      .replace("找片大师定风波", "")
      .replace("@公众号", "")
      .replace("高清", "")
      .replace("犯罪惊悚", "")
      .replace("喜剧动作", "")
      .replace("动作", "")
      .replace("奇幻", "")
      .replace("印地语", "")
      .replace("加长版", "")
      .replace("远鉴", "")
      .replace("字幕组", "")
      .replace("精校中英字幕", " ")
      .replace("真人电影", " ")
      .replace("超前完结", " ")
      .replace("番外", " ")
      .replace("彩蛋", " ")
      .replace("豆瓣", " ")
      .replace("完整全集", " ")
      .replace("稀有国日双语版", " ")
      .replace("官方中英双字", " ")
      .replace("(CC版)", " ")
      .replace("国粤英多音轨", " ")
      .replace("粤语音轨", " ")
      .replace("国英双语", " ")
      .replace("国语配音版", " ")
      .replace("韩语中字", " ")
      .replace("韩语官中", " ")
      .replace("字幕版", " ")
      .replace("国配简繁特效", " ")
      .replace("简繁双语特效字幕", " ")
      .replace("导评简繁六字幕", " ")
      .replace("内封简英双字", " ")
      .replace("内封简、繁中字", " ")
      .replace("国粤英3语", " ")
      .replace("国粤语配音", " ")
      .replace("粤语配音", " ")
      .replace("中英双语字幕", " ")
      .replace("带中文字幕", " ")
      .replace("日英四语", " ")
      .replace("国日双语", " ")
      .replace("官方中字", " ")
      .replace("台日双语", " ")
      .replace("华语配音", " ")
      .replace("普通话版", " ")
      .replace("（普通话）", " ")
      .replace("外挂双语", " ")
      .replace("内封中字", " ")
      .replace("有字幕", " ")
      .replace("双语版", " ")
      .replace("日语版", " ")
      .replace("电视版本", " ")
      .replace("电视版", " ")
      .replace("外挂中文字幕", " ")
      .replace("内封多字幕", " ")
      .replace("中英特效字幕", " ")
      .replace("简繁英双语字幕", " ")
      .replace("简繁英特效字幕", " ")
      .replace("简繁英双语特效字幕", " ")
      .replace("简繁英字幕", " ")
      .replace("简体字幕", " ")
      .replace("繁英字幕", " ")
      .replace("简繁字幕", " ")
      .replace("繁英字幕", " ")
      .replace("简英字幕", " ")
      .replace("简繁双语字幕", " ")
      .replace("国语音轨", " ")
      .replace("国语配音", " ")
      .replace("国韩多音轨", " ")
      .replace("国英多音轨", " ")
      .replace("多音轨", " ")
      .replace("(粤语中字)", " ")
      .replace("英语中字", " ")
      .replace("BD中英双字", " ")
      .replace("特效中英双字", " ")
      .replace("中英双字", " ")
      .replace("中英字幕", " ")
      .replace("特效字幕", " ")
      .replace("中文字幕", " ")
      .replace("日语无字", " ")
      .replace("国日英三语", " ")
      .replace("日粤英三语", " ")
      .replace("简日双语内封", " ")
      .replace("陆台粤日语", " ")
      .replace("简繁日内封", " ")
      .replace("粤日中字", " ")
      .replace("台配繁中", " ")
      .replace("简中内封", " ")
      .replace("简中内嵌", " ")
      .replace("简体内嵌", " ")
      .replace("简繁内嵌", " ")
      .replace("简繁内嵌", " ")
      .replace("简繁内封", " ")
      .replace("无字幕", " ")
      .replace("双语", " ")
      .replace("国语版", " ")
      .replace("国语", " ")
      .replace("国英", " ")
      .replace("中配", " ")
      .replace("官中", " ")
      .replace("粤语", " ")
      .replace("国粤", " ")
      .replace("西班牙语", " ")
      .replace("剧情", " ")
      .replace("仅英轨", " ")
      .replace("配音版", " ")
      .replace("台配国语", " ")
      .replace("台配", " ")
      .replace("俄语", " ")
      .replace("泰语", " ")
      .replace(".中日双语", " ")
      .replace(".日语", " ")
      .replace("日语", " ")
      .replace("英语", " ")
      .replace("日语版", " ")
      .replace("全系列电影", " ")
      .replace("系列合集", " ")
      .replace("大合集", " ")
      .replace("合集", " ")
      .replace("-系列", " ")
      .replace("系列", " ")
      .replace("持续更新中", " ")
      .replace("更新中", " ")
      .replace(".内嵌", " ")
      .replace(".日配", " ")
      .replace("(客串)", " ")
      .replace("HD720P", " ")
      .replace("720P", " ")
      .replace(".720p", " ")
      .replace(".720P", " ")
      .replace("HD720P", " ")
      .replace("720P", " ")
      .replace(".1080p", " ")
      .replace(".1080P", " ")
      .replace("HD1080P", " ")
      .replace("1080p", " ")
      .replace("1080P", " ")
      .replace(".2160p", " ")
      .replace("2160p", " ")
      .replace("2160P", " ")
      .replace("3840x2160", " ")
      .replace("120帧率版本", " ")
      .replace("60FPS修复珍藏版", " ")
      .replace("60帧率版本", " ")
      .replace("音轨版", " ")
      .replace("HDR版本", " ")
      .replace("[HDR]", " ")
      .replace("HDR", " ")
      .replace("MP4", " ")
      .replace(".4k", " ")
      .replace(" 4k ", " ")
      .replace("高码4K", " ")
      .replace("4K修复版", " ")
      .replace("4K修复", " ")
      .replace("蓝光原盘REMUX", " ")
      .replace("4K原盘REMUX", " ")
      .replace("4K REMUX", " ")
      .replace("杜比视界", " ")
      .replace("杜比", " ")
      .replace("全景声版", " ")
      .replace("REMUX", " ")
      .replace("REMXU", " ")
      .replace("RMVB", " ")
      .replace("4K HDR", " ")
      .replace("4K版", " ")
      .replace("纯净版", " ")
      .replace("10bit", " ")
      .replace("60fps", " ")
      .replace("WEB-DL", " ")
      .replace("BD", " ")
      .replace("DDP5", " ")
      .replace("BluRay", " ")
      .replace("H265", " ")
      .replace("H264", " ")
      .replace("x265", " ")
      .replace("X264", " ")
      .replace("x264", " ")
      .replace("4K修复珍藏版", " ")
      .replace("蓝光原盘", " ")
      .replace("蓝光高清", " ")
      .replace("蓝光版", " ")
      .replace("蓝光", " ")
      .replace("高码版", " ")
      .replace("部分高清", " ")
      .replace("标清", " ")
      .replace("4K原盘", " ")
      .replace("超清4K修复", " ")
      .replace("超清", " ")
      .replace("4K修复版", " ")
      .replace("4K收藏版", " ")
      .replace("4K双版本", " ")
      .replace("最终剪辑版", " ")
      .replace("双版本", " ")
      .replace("[4K]", " ")
      .replace("4K", " ")
      .replace("4k", " ")
      .replace("60帧", " ")
      .replace("高码率", " ")
      .replace(".超高码率", " ")
      .replace("杜比视界版本", " ")
      .replace("IMAX", " ")
      .replace("+外传", " ")
      .replace("+番外篇", " ")
      .replace("+番外", " ")
      .replace("+漫画", " ")
      .replace("+电影", " ")
      .replace("国漫-", " ")
      .replace("电视剧", " ")
      .replace("剧版", " ")
      .replace("网剧", " ")
      .replace("短剧", " ")
      .replace("衍生剧", " ")
      .replace("美漫", " ")
      .replace("全季", " ")
      .replace("加剧场版", " ")
      .replace("+剧场版", " ")
      .replace("剧场版", " ")
      .replace("加外传", " ")
      .replace("+真人版", " ")
      .replace("真人版", " ")
      .replace("精编版", " ")
      .replace("电视系列片", " ")
      .replace("纪录片专场", " ")
      .replace("真实人物改编", " ")
      .replace("真实故事", " ")
      .replace("迷你剧", " ")
      .replace("系列片", " ")
      .replace("动漫加真人", " ")
      .replace("动漫+真人", " ")
      .replace("导演剪辑版", " ")
      .replace("高码收藏版", " ")
      .replace("高码", " ")
      .replace("高清黑金珍藏版", " ")
      .replace("高清修复版", " ")
      .replace("重置版", " ")
      .replace("洗版", " ")
      .replace("特典映像", " ")
      .replace("收藏版", " ")
      .replace("「珍藏版」", " ")
      .replace("珍藏版", " ")
      .replace("极致版", " ")
      .replace("典藏版", " ")
      .replace("特别版", " ")
      .replace("老版", " ")
      .replace("经典老剧", " ")
      .replace("经典剧", " ")
      .replace("连续剧", " ")
      .replace("未删减版", " ")
      .replace("未删减", " ")
      .replace("无删减", " ")
      .replace("无台标", " ")
      .replace("重制版", " ")
      .replace("完整高清", " ")
      .replace("完结篇", " ")
      .replace("完结", " ")
      .replace("高分剧", " ")
      .replace("未精校", " ")
      .replace("霸王龙压制", " ")
      .replace("酷漫字幕组", " ")
      .replace("凤凰天使", " ")
      .replace("[一只鱼4kyu.cc]", " ")
      .replace("（流媒体）", " ")
      .replace("+Q版", " ")
      .replace("+OVA", " ")
      .replace("+SP", " ")
      .replace("+前传", " ")
      .replace("中国大陆区", " ")
      //.replace("大陆", " ")
      .replace("未分级重剪加长版", " ")
      .replace("【美剧】", " ")
      .replace("【法国】", " ")
      .replace("【西班牙】", " ")
      .replace("【俄罗斯】", " ")
      .replace("【英剧】", " ")
      .replace("【爱情片】", " ")
      .replace("【纪录片】", " ")
      .replace("泰国奇幻剧", " ")
      .replace("【美漫】", " ")
      .replace("美剧", " ")
      .replace("喜剧爱情", " ")
      .replace("喜剧", " ")
      .replace("综艺", " ")
      .replace("意大利", " ")
      .replace("恐怖剧", " ")
      .replace("科幻剧", " ")
      .replace("国产剧", " ")
      .replace("国产", " ")
      .replace("悬疑|传记剧", " ")
      .replace("悬疑", " ")
      .replace("[恐怖]", " ")
      .replace("惊悚", " ")
      .replace("短片", " ")
      .replace("电影版", " ")
      .replace("-系列", " ")
      .replace("系列", " ")
      .replace("全集", " ")
      .replace("中字", " ")
      .replace("外挂字幕", " ")
      .replace("字幕", " ")
      .replace("无字", " ")
      .replace("无水印版", " ")
      .replace("无水印", " ")
      .replace("腾讯水印", " ")
      .replace("腾讯", " ")
      .replace("B站", " ")
      .replace("OVA", " ")
      .replace("TV加MOV", " ")
      .replace("HDTV", " ")
      .replace("GOTV", " ")
      .replace("NHK", " ")
      .replace("人人影视制作", " ")
      .replace("高清翡翠台", " ")
      .replace("TVB版", " ")
      .replace("TVB", " ")
      .replace("ATV", " ")
      .replace("BBC", " ")
      .replace("(剧版)", " ")
      .replace("DVD版", " ")
      .replace("DVD", " ")
      .replace("《单片》", " ")
      .replace("公众号：锦技社", " ")
      .replace("推荐!", " ")
      .replace("[", " ")
      .replace("]", " ")
      .replace("【", " ")
      .replace("】", " ")
      .replace("（", "(")
      .replace("）", ")")
      .replace("《", " ")
      .replace("》", " ")
      .replace(",", " ")
      .replace("..", " ")
      .replace("_", " ")
      .replace("⭐", " ")
      .replace("|", " ")
      .replace("+", " ")
      .replace("III", "第三季")
      .replace("II", "第二季")
      .replace("Ⅱ", "第二季")
      .replace("Ⅲ", "第三季")
      .replace("Ⅳ", "第四季")
      .replace("Ⅴ", "第五季")
      .replace("Ⅵ", "第六季")
      .replace("~", " ")
      .replace("双字", " ")
      .replace("未分级版", " ")
      .replace("内封", " ")
      .replace("纪录片", " ")
      .replace("更多", " ")
      .replace("资源", " ");

    let m = TextUtils.NAME1.exec(newName);
    if (m) {
      newName = m[1];
    }

    m = TextUtils.NAME2.exec(newName);
    if (m) {
      newName = m[1];
    }

    m = TextUtils.NAME3.exec(newName);
    if (m) {
      newName = m[1];
    }

    //如果文件名只剩下集数就没用了
    if (/^[SE0-9]+$/.test(newName)) {
      newName = "";
    }
    cfLog("pos3", newName);

    newName = newName
      .replace(/No.\d+ ?/g, " ")
      .replace(/\d+、/g, " ")
      .replace(/\.\d{4}/g, " ")
      .replace(/ \d{4}/g, " ")
      .replace(/\s*全\d+集/g, " ")
      .replace(/第?\d-\d+([季部])/g, " ")
      .replace(/.([季部])全/g, " ")
      .replace(/[0-9.]+GB/g, " ")
      .replace(/豆瓣评分：?[0-9.]+/g, " ")
      .replace(/NO \d+\｜/g, " ")
      .replace(/\(\d{4}\)/g, " ")
      .replace(/\.\d+集全/g, " ");

    cfLog("pos4", newName);
    m = TextUtils.NUMBER.exec(newName);
    if (m) {
      let text = m[1];
      if (m.index > 1 && newName.charAt(m.index - 1) != ' ') {
        newName = newName.replace("第" + text + "季", " 第" + text + "季");
      }
      let newNum = TextUtils.number2text(text);
      newName = newName.replace(text, newNum);
    } else {
      m = TextUtils.NUMBER2.exec(newName);
      if (m) {
        let text = m[1];
        if (m.index > 0 && newName.charAt(m.index - 1) != ' ') {
          newName = newName.replace(text, " " + text);
        }
      } else {
        m = TextUtils.NUMBER3.exec(newName);
        if (m) {
          let text = m[1];
          let newNum = TextUtils.number2text(text.substring(1));
          newName = newName.replace(text, " 第" + newNum + "季");
        }
      }
    }
    cfLog("pos5", newName);

    m = TextUtils.NUMBER4.exec(newName);
    if (m) {
      let text = m[1];
      newName = newName.replace(text, "");
    } else {
      m = TextUtils.NUMBER5.exec(newName);
      if (m) {
        let text = m[1];
        newName = newName.replace(text, "");
      } else {
        m = TextUtils.NUMBER6.exec(newName);
        if (m) {
          let text = m[1];
          newName = newName.replace(text, "");
        }
      }
    }

    cfLog("pos6", newName);
    newName = newName
      .replace(/\./g, " ")
      .replace(/\s+/g, " ")
      .trim();
    newName = TextUtils.removeEnglish(newName)
      .replace(/\s+/g, " ")
      .trim();

    //去掉独立的数字，多半是删除英文后残留的
    newName = newName
      .replace(/\s/g, "  ")
      .replace(/\s\d+\s/g, " ")
      .replace(/\s\d+/g, " ")
      .replace(/^\d+\s/g, " ")
      .replace(/\s+/g, " ")
      .trim();

    //如果是中文和其它语言混合，仅把中文提取出来
    parts = newName.split(' ');
    let chineseParts = [];
    if (parts.length > 0) {
      parts.forEach(element => {
        if (TextUtils.isChinese(element)) {
          chineseParts.push(element);
        }
      });
    }
    if (chineseParts.length > 0) {
      newName = chineseParts.join(' ');
    }

    return newName;
  }

  static number2text(text) {
    if (text.startsWith("0") && text.length > 1) {
      text = text.substring(1);
    }
    if (!text) {
      return text;
    }
    let num = parseInt(text);
    let newNum;
    if (num <= 10) {
      newNum = TextUtils.NUMBERS[num];
    } else if (num < 20) {
      newNum = "十" + TextUtils.NUMBERS[num % 10];
    } else if (num % 10 === 0) {
      newNum = TextUtils.NUMBERS[num / 10] + "十";
    } else {
      newNum = TextUtils.NUMBERS[Math.floor(num / 10)] + "十" + TextUtils.NUMBERS[num % 10];
    }
    return newNum;
  }

  static updateName(name) {
    let n = name.length;
    if (n > 1 && (TextUtils.isEnglishChar(name.codePointAt(0)) && TextUtils.isChineseChar(name.codePointAt(1)))) {
      name = name.substring(1);
      n = name.length;

    }

    if (n > 2) {
      if (TextUtils.isEnglishChar(name.codePointAt(0))
        && (name.charAt(1) === ' ' || name.charAt(1) === '.')
        && TextUtils.isChineseChar(name.codePointAt(2))) {
        name = name.substring(2);
        n = name.length;
      }

      if (name.charAt(n - 1) === '1' && TextUtils.isChineseChar(name.codePointAt(n - 2))) {
        name = name.substring(0, n - 1);
      }
    }

    if (name.endsWith(" 第一季")) {
      name = name.substring(0, name.length - 4);
    }

    let start = name.indexOf('.');
    if (start === 4) {
      try {
        parseInt(name.substring(0, 4));
        let end = name.indexOf('.', start + 1);
        if (end > start + 1) {
          name = name.substring(start + 1, end);
        }
      } catch (e) {
        // ignore
      }
    }
    return name;
  }

  static truncate(charSequence, threshold) {
    return charSequence.length > threshold ? charSequence.subSequence(0, threshold) + "..." : charSequence.toString();
  }

  static removeEnglish(str) {
    if (/^[a-zA-Z!"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~]+$/.test(str)) {
      return str; // 如果字符串是纯英文，则直接返回
    } else {
      return str
        .replace(/[a-zA-Z!"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~]/g, '');// 否则，去掉英文部分和标点符号
    }
  }

  static removeEnglishExtension(filename) {
    const regex = /\.[a-zA-Z0-9]+$/;
    return filename.replace(regex, '');
  }

  static similarity(sourceStr, targetStr) {
    sourceStr = sourceStr.replace(/\s/g, '');
    targetStr = targetStr.replace(/\s/g, '');
    sourceStr = sourceStr
      .replace(/第一季/g, "一")
      .replace(/第二季/g, "二")
      .replace(/第三季/g, "三")
      .replace(/第四季/g, "四")
      .replace(/第五季/g, "五")
      .replace(/第六季/g, "六")
      .replace(/第七季/g, "七")
      .replace(/第八季/g, "八")
      .replace(/第九季/g, "九")
      .replace(/0?1/g, "一")
      .replace(/0?2/g, "二")
      .replace(/0?3/g, "三")
      .replace(/0?4/g, "四")
      .replace(/0?5/g, "五")
      .replace(/0?6/g, "六")
      .replace(/0?7/g, "七")
      .replace(/0?8/g, "八")
      .replace(/0?9/g, "九");
    targetStr = targetStr
      .replace(/第一季/g, "一")
      .replace(/第二季/g, "二")
      .replace(/第三季/g, "三")
      .replace(/第四季/g, "四")
      .replace(/第五季/g, "五")
      .replace(/第六季/g, "六")
      .replace(/第七季/g, "七")
      .replace(/第八季/g, "八")
      .replace(/第九季/g, "九")
      .replace(/0?1/g, "一")
      .replace(/0?2/g, "二")
      .replace(/0?3/g, "三")
      .replace(/0?4/g, "四")
      .replace(/0?5/g, "五")
      .replace(/0?6/g, "六")
      .replace(/0?7/g, "七")
      .replace(/0?8/g, "八")
      .replace(/0?9/g, "九");


    sourceStr = TextUtils.removeEnglish(sourceStr).replace(/\s/g, "");
    targetStr = TextUtils.removeEnglish(targetStr).replace(/\s/g, "");

    cfLog(sourceStr, targetStr, sourceStr === targetStr);

    let sourceLen = sourceStr.length;
    let targetLen = targetStr.length;

    if (sourceLen === 0) {
      return targetLen;
    }
    if (targetLen === 0) {
      return sourceLen;
    }

    let arr = Array.from({ length: sourceLen + 1 }, () => Array(targetLen + 1).fill(0));

    for (let i = 0; i < sourceLen + 1; i++) {
      arr[i][0] = i;
    }

    for (let j = 0; j < targetLen + 1; j++) {
      arr[0][j] = j;
    }

    let sourceChar;
    let targetChar;

    for (let i = 1; i < sourceLen + 1; i++) {
      sourceChar = sourceStr.charAt(i - 1);
      for (let j = 1; j < targetLen + 1; j++) {
        targetChar = targetStr.charAt(j - 1);
        if (sourceChar === targetChar) {
          arr[i][j] = arr[i - 1][j - 1];
        } else {
          arr[i][j] = (Math.min(Math.min(arr[i - 1][j], arr[i][j - 1]), arr[i - 1][j - 1])) + 1;
        }
      }
    }

    return 1 - arr[sourceLen][targetLen] / Math.max(sourceLen, targetLen);
  }

  static isNormal(name) {
    let n = name.length;
    if (n === 0) {
      return false;
    }
    if (name.toLowerCase().startsWith("season ")) {
      return false;
    }
    if (name.includes("花絮")) {
      return false;
    }
    if (name.includes("彩蛋")) {
      return false;
    }
    if (name === "高画质版") {
      return false;
    }
    if (name === "国配") {
      return false;
    }
    if (name === "高码") {
      return false;
    }
    if (name === "SDR") {
      return false;
    }
    if (name === "A-Z") {
      return false;
    }
    if (name === "02") {
      return false;
    }
    if (name === "11 - à Zélie") {
      return false;
    }
    if (name === "CO(rrespondance) VID(éo) #9 - à Zélie") {
      return false;
    }
    if (name === "CO(rrespondance) VID(éo) #11 - à Zélie") {
      return false;
    }
    if (name === "字幕") {
      return false;
    }
    if (name === "真人秀") {
      return false;
    }
    if (name === "曲艺辙痕") {
      return false;
    }
    if (name === "字幕勿扰") {
      return false;
    }
    if (name === "新奇的整理") {
      return false;
    }
    if (name === "读·豆瓣") {
      return false;
    }
    if (name === "每个人都有他自己的电影") {
      return false;
    }
    if (name === "SP") {
      return false;
    }
    if (name === "TV") {
      return false;
    }
    if (name === "OAD") {
      return false;
    }
    if (name === "actors") {
      return false;
    }
    if (name === "动漫") {
      return false;
    }
    if (name === "特别篇") {
      return false;
    }
    if (name === "中国") {
      return false;
    }
    if (name === "韩国") {
      return false;
    }
    if (name === "意大利") {
      return false;
    }
    if (name === "澳大利亚") {
      return false;
    }
    if (name === "非洲") {
      return false;
    }
    if (name === "新建文件夹") {
      return false;
    }
    if (name === "Specials") {
      return false;
    }
    if (name === "TV字幕") {
      return false;
    }
    if (name === "剧场版") {
      return false;
    }
    if (name === "大合集") {
      return false;
    }
    if (name === "蓝光电影") {
      return false;
    }
    if (name === "电影版") {
      return false;
    }
    if (name.toLowerCase() === "movie") {
      return false;
    }
    if (name.toLowerCase() === "ost") {
      return false;
    }
    if (name.toLowerCase() === "2160p") {
      return false;
    }
    if (name === "番外") {
      return false;
    }
    if (name === "国语版") {
      return false;
    }
    if (/^\d+$/.test(name)) {
      return false;
    }
    if (name.startsWith("S") && /^\d+$/.test(name.substring(1))) {
      return false;
    }
    if (name.toUpperCase().startsWith("1080P")) {
      return false;
    }
    if (name.toUpperCase().startsWith("4K")) {
      return false;
    }
    if (name.endsWith("版本")) {
      return false;
    }
    if (name.endsWith("语版")) {
      return false;
    }
    if (name.endsWith(" 番外")) {
      return false;
    }
    if (name.endsWith(" 大电影")) {
      return false;
    }
    if (n === 1 && TextUtils.isEnglishChar(name.charAt(0))) {
      return false;
    }
    return true;
  }

  static isSpecialFolder(name) {
    if (name.toLowerCase().startsWith("4k")) {
      return true;
    }
    if (name.toLowerCase().startsWith("2160p")) {
      return true;
    }
    if (name.toLowerCase().startsWith("1080p")) {
      return true;
    }
    if (name === "SDR") {
      return true;
    }
    if (name === "国语") {
      return true;
    }
    if (name === "国语版") {
      return true;
    }
    if (name === "粤语") {
      return true;
    }
    if (name === "粤语版") {
      return true;
    }
    if (name === "番外彩蛋") {
      return true;
    }
    if (name === "彩蛋") {
      return true;
    }
    if (name === "付费花絮合集") {
      return true;
    }
    if (name === "大结局点映礼") {
      return true;
    }
    if (name === "心动记录+彩蛋") {
      return true;
    }
    return false;
  }

  static getName(path) {
    const index = path.lastIndexOf('/');
    if (index > -1) {
      return path.substring(index + 1);
    }
    return path;
  }

  static getParentName(path) {
    const index = path.lastIndexOf('/');
    if (index > -1) {
      path = path.substring(0, index);
    }
    return TextUtils.getName(path);
  }

  static getParent(path) {
    const index = path.lastIndexOf('/');
    if (index > 0) {
      return path.substring(0, index);
    }
    return "";
  }

  static getYearFromPath(path) {
    const max = new Date().getFullYear() + 3;
    const parts = path.split("/");
    for (let i = parts.length - 1; i >= 0; i--) {
      const m = parts[i].match(/\b(\d{4})\b/);
      if (m) {
        const year = parseInt(m[1]);
        if (year > 1960 && year < max) {
          console.debug(`find year ${year} from path ${path}`);
          return year;
        }
      }
    }
    return null;
  }

  static getNameFromPath(path) {
    let name = TextUtils.getName(path);
    if (TextUtils.isSpecialFolder(name)) {
      name = TextUtils.getParentName(name);
    }

    let parts = name.split("|");
    if (parts.length > 3) {
      name = parts[0];
    }
    const NUMBER = /Season (\d{1,2})/;
    const NUMBER2 = /SE(\d{1,2})/;
    const NUMBER3 = /^S(\d{1,2})$/;
    const NUMBER1 = /第(\d{1,2})季/;
    if (name.startsWith("Season ")) {
      const m = name.match(NUMBER);
      if (m) {
        const text = m[1];
        const newNum = TextUtils.number2text(text);
        name = TextUtils.fixName(TextUtils.getParentName(path)) + " 第" + newNum + "季";
      }
    } else if (name.startsWith("第")) {
      const m = name.match(NUMBER1);
      if (m) {
        const text = m[1];
        const newNum = TextUtils.number2text(text);
        name = TextUtils.fixName(TextUtils.getParentName(path)) + " 第" + newNum + "季";
      } else if (name.endsWith("季")) {
        name = TextUtils.fixName(TextUtils.getParentName(path)) + " " + name;
      }
    } else if (name.startsWith("SE")) {
      const m = name.match(NUMBER2);
      if (m) {
        const text = m[1];
        const newNum = TextUtils.number2text(text);
        name = TextUtils.fixName(TextUtils.getParentName(path)) + " 第" + newNum + "季";
      }
    } else if (name.startsWith("S")) {
      const m = name.match(NUMBER3);
      if (m) {
        const text = m[1];
        const newNum = TextUtils.number2text(text);
        name = TextUtils.fixName(TextUtils.getParentName(path)) + " 第" + newNum + "季";
      }
    }

    name = TextUtils.fixName(name);
    let orgPath = path;
    cfLog("pos2", name);
    if (!TextUtils.isNormal(name)) {
      path = TextUtils.getParent(path);
      if (path) {
        let res = TextUtils.getNameFromPath(path);
        if (res) {
          res.org_path = orgPath;
        }
        return res;
      }
    }
    else {
      return { name: name, index_path: path, org_path: orgPath };
    }
  }
}

async function fetchOverTcp(request) {
  let url = new URL(request.url);
  let req = new Request(url, request);
  let port_string = url.port;
  if (!port_string) {
    port_string = url.protocol === "http:" ? "80" : "443";
  }
  let port = parseInt(port_string);

  /*
  if ((url.protocol === "https:" && port === 443) || (url.protocol === "http:" && port === 80)) {
    // CF标准的反代不支持IP地址，所以IPV6要走TCP代理
    if (!isIP(url.host)) {
      return await fetch(req);
    }
  }
  */

  // 创建 TCP 连接
  let tcpSocket = connect({
    hostname: url.hostname,
    port: port,
  }, JSON.parse('{"secureTransport": "starttls"}'));

  if (url.protocol === "https:") {
    tcpSocket = tcpSocket.startTls();
  }

  try {
    const writer = tcpSocket.writable.getWriter();

    // 构造请求头部
    let headersString = '';
    let bodyString = '';

    for (let [name, value] of req.headers) {
      //if (name !== "accept-encoding" && name !== "host") {
      if (name === "user-agent" || name === "accept") {
        headersString += `${name}: ${value}\r\n`;
      }

    }
    headersString += `connection: close\r\n`;
    headersString += `accept-encoding: identity\r\n`;
    headersString += `host: ${url.hostname}\r\n`

    let fullpath = url.pathname;

    // 如果有查询参数，将其添加到路径
    if (url.search) {
      fullpath += url.search.replace(/%3F/g, "?");
    }

    const body = await req.text();
    bodyString = `${body}`;

    let reqText = `${req.method} ${fullpath} HTTP/1.1\r\n${headersString}\r\n${bodyString}\r\n`;
    console.log("Raw Request", reqText);
    let reqBuff = new TextEncoder().encode(reqText);


    //return new Response(reqBuff);

    // 发送请求
    await writer.write(reqBuff);
    writer.releaseLock();

    // 获取响应
    const response = await constructHttpResponse(tcpSocket);

    console.log("fetchOverTcp response headers", JSON.parse(JSON.stringify(Object.fromEntries(Array.from(response.headers.entries())))));

    return response;
  } catch (error) {
    console.log("fetchOverTcp Exception", error);
    tcpSocket.close();
    return new Response(error.stack, { status: 500 });
  }
}

async function constructHttpResponse(tcpSocket, timeout) {
  const reader = tcpSocket.readable.getReader();
  let remainingData = new Uint8Array(0);
  try {
    // 读取响应数据
    while (true) {
      const { value, done } = await reader.read();
      const newData = new Uint8Array(remainingData.length + value.length);
      newData.set(remainingData);
      newData.set(value, remainingData.length);
      remainingData = newData;
      const index = indexOfDoubleCRLF(remainingData);
      if (index !== -1) {
        reader.releaseLock();
        const headerBytes = remainingData.subarray(0, index);
        const bodyBytes = remainingData.subarray(index + 4);

        const header = new TextDecoder().decode(headerBytes);
        const [statusLine, ...headers] = header.split('\r\n');
        const [httpVersion, statusCode, ...tmpStatusText] = statusLine.split(' ');
        let statusText = tmpStatusText.join(' ');

        // 构造 Response 对象
        let responseHeaders = JSON.parse('{}');
        headers.forEach((header) => {
          const [name, value] = header.split(': ');
          responseHeaders[name.toLowerCase()] = value;
        });

        responseHeaders = JSON.parse(JSON.stringify(responseHeaders));
        console.log("orginal responseHeaders", responseHeaders);

        const responseInit = {
          status: parseInt(statusCode),
          statusText,
          headers: new Headers(responseHeaders),
        };

        console.log("statusCode", statusCode);

        let readable = null;
        let writable = null;
        let stream = null;
        if (responseHeaders["content-length"]) {
          stream = new FixedLengthStream(parseInt(responseHeaders["content-length"]));
        } else {
          stream = new TransformStream();
        }
        readable = stream.readable;
        writable = stream.writable;

        //规避CF问题，延迟1ms执行
        function delayedExecution() {
          setTimeout(() => {
            let writer = writable.getWriter();
            writer.write(bodyBytes);
            writer.releaseLock();
            tcpSocket.readable.pipeTo(writable);
          }, 1);
        }
        delayedExecution();

        return new Response(readable, responseInit);
      }
      if (done) {
        tcpSocket.close();
        break;
      }
    }

    console.log("Response Done!");
    return new Response();
  } catch (error) {
    console.log("Construct Response Exception", error);
    tcpSocket.close();
  }
}

function indexOfDoubleCRLF(data) {
  if (data.length < 4) {
    return -1;
  }
  for (let i = 0; i < data.length - 3; i++) {
    if (data[i] === 13 && data[i + 1] === 10 && data[i + 2] === 13 && data[i + 3] === 10) {
      return i;
    }
  }
  return -1;
}

async function getDoubanInfo(id) {
  try {
    const url = `https://movie.douban.com/subject/${id}/`;
    const response = await fetch(url);
    const html = await response.text();

    // 使用正则表达式解析HTML内容
    // 1. 解析剧情简介
    const plotMatch = html.match(/<span property="v:summary"[^>]*>([\s\S]*?)<\/span>/);
    const plot = plotMatch ? plotMatch[1].replace(/<[^>]+>/g, '').trim() : '';

    // 2. 解析年份
    const year = html.match(/<span class="year">[（(]?\s*(\d{4})\s*[）)]?<\/span>/)?.[1] || '';

    // 3. 解析国家地区
    const regionMatch = html.match(/<span class="pl">制片国家\/地区:<\/span>[\s\n]*([^<]+)/);
    const region = regionMatch ? regionMatch[1].trim() : '';

    // 4. 解析演员列表
    const actorRegex = /<meta property="video:actor" content="([^"]+)"\/?>/g;
    const actors = [];
    let actorMatch;
    while ((actorMatch = actorRegex.exec(html)) !== null) {
      actors.push(actorMatch[1]);
    }

    // 5. 解析导演
    const directorMatch = html.match(/<meta property="video:director" content="([^"]+)"\/?>/);
    const director = directorMatch ? directorMatch[1] : '';

    // 6. 解析类型
    const typeRegex = /<span property="v:genre">([^<]+)<\/span>/g;
    const types = [];
    let typeMatch;
    while ((typeMatch = typeRegex.exec(html)) !== null) {
      types.push(typeMatch[1]);
    }

    // 7. 解析评分
    const ratingMatch = html.match(/<strong class="ll rating_num" property="v:average">([^<]+)<\/strong>/);
    const rating = ratingMatch ? ratingMatch[1].trim() : '';

    // 返回解析结果
    return {
      //plot: plot,
      year: year,
      region: region,
      //actors: actors.join('/'),
      //director: director,
      type: types.join('/'),
      //rating: rating
    };
  } catch (error) {
    return {};
  }
}

let keyword = "";
async function handleRequest(request) {
  let orgUrl = new URL(request.url);
  if (orgUrl.pathname.startsWith("/search")) {
    //let googUrl = 'https://www.google.com.hk/search?q=' + orgUrl.searchParams.get('q') + '&num=100';
    //let googUrl = 'https://www.google.com.hk/search' + orgUrl.search;
    let googUrl = 'https://www.bing.com/search' + orgUrl.search + "&mkt=zh-CN";
    //let googUrl = 'https://www.bing.com/search' + orgUrl.search; 
    let newUrl = new URL(googUrl)
    let method = request.method
    let request_headers = request.headers
    let new_request_headers = new Headers(request_headers)

    //new_request_headers.set('Host', "www.google.com.hk")
    //new_request_headers.set('Referer', "https://www.google.com.hk")

    let newReq = new Request(newUrl, {
      method: method,
      headers: new_request_headers,
    });
    let original_response = await fetchOverTcp(newReq);

    return original_response;
  } else if (orgUrl.pathname.startsWith("/doubaninfo")) {
    try {
      // 调用豆瓣信息获取函数
      const params = orgUrl.searchParams;
      const id = params.get('id');
      const info = await getDoubanInfo(id);
      return new Response(JSON.stringify(info, null, 2), {
        headers: { 'Content-Type': 'application/json' }
      });
    } catch (error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  }

  keyword = decodeURIComponent(orgUrl.searchParams.get('q'));
  let name = TextUtils.getNameFromPath(keyword).name;

  keyword = name;
  const url = "https://m.douban.com/search/?type=movie&query=" + encodeURIComponent(name);
  const response = await fetch(url);
  let result = await transformHTMLtoJSON(await response.text(), ["a", "img"]);

  //return new Response(JSON.stringify(result, null, 2));

  result = formatDoubanJson(result);
  
  const promises = result.map(async (item) => {
    let info = {};
    if (item.id) {
      try {
        if (item.similarity && +item.similarity >= 0.8) {
          info = await getDoubanInfo(item.id);
        } else {
          info = {};
        }
        
      } catch (error) {
        info = {};
      }
    }
    return Object.assign({}, item, info);
  });
  
  result = await Promise.all(promises);

  return new Response(JSON.stringify(result, null, 2));
}

function parsePath(items) {
  let result = [];
  let parts = items.split('^');
  parts.forEach(element => {
    result.push(TextUtils.getNameFromPath(element));
  });
  return result;
}

(async () => {
  /*
  if (typeof addEventListener === "function") {
    return
  }
  */
  //For Nodejs, to resuse some js API
  isInCf = false;
  let act = "";
  eval('act = process.argv[2]');
  let result = eval(act);
  console.log(JSON.stringify(result, null, 2));
})();
