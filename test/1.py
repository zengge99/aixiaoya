from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
"/我的115分享/电影/原盘/合集/06 高码率视频(电影为主，少量纪录片剧集演唱会)MKV REMUX格式为主，多为单片20G以上，初整理，不重，多版本除外) 437.92T 15937个文件(若干外挂字幕)/00画质控蓝光原盘REMUX 53.45T 2694/3. 电影-蓝光原盘REMUX/九龙城寨之围城 蓝光原盘REMUX [国粤双语] [中英双字 国配中字 官译中字]/Twilight.Of.The.Warriors.Walled.In.2024.1080p.BluRay.REMUX.AVC.DTS-HD.MA.TrueHD.7.1.Atmos.mkv",
candidate_labels=["movie", "education", "politics", "business"],
)
print(result)