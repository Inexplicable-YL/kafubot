from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import opencc
from langchain.tools import tool

from kafubot.cognition.plugins.base import PluginContext, PluginDefinition, ToolScope

KAFU_SONGS_PROMPT = "以下是根据`{text}`搜索到的相关歌曲：\n"
SONGS_LIST = [
    "「Angel（feat.可不）」 - 飽海",
    "「ツキミチシルベ [vocaloid ver.]（feat.可不 & 初音ミク）」 - MIMI",
    "「Boi（feat.可不）」 - ポリスピカデリー",
    "「化孵化（feat.可不）」 - sasakure.UK",
    "「VIPエンジョイ（feat.可不 & 星界 & 裏命 & 羽累 & 狐子）」 - V.I.P",
    "「ラブマシーン。（feat.可不）」 - A4。",
    "「その銃口（feat.可不）」 - あばらや",
    "「CH4NGE（feat.可不）」 - Giga",
    "「マーシャル・マキシマイザー（feat.可不）」 - 柊マグネタイト（柊磁鐵）",
    "「サムワンズノウ - 加納先生（feat.可不）」 - 包丁ナイフカッターズ",
    "「しみったれ（feat.可不）」 - やんかな",
    "「デイバイデイズ（feat.可不 & 初音ミク）」 - syudou",
    "「幽玄の詩（feat.可不 & 星界 & 裏命）」 - ◇*ゆくえわっと",
    "「東京ナイトホーク（feat.可不）」 - 夏山よつぎ",
    "「キッカイケッタイ（feat.可不 & 初音ミク）」 - メドミア",
    "「しあわせのはこ（feat.可不 & 星界 & ナースロボ_タイプT）」 - kyiku",
    "「なりきれない（feat.可不）」 - 雨良 Amala",
    "「脆弱性（feat.可不）」 - にほしか",
    "「絶対敵対メチャキライヤー（feat.可不 & 初音ミク）」 - メドミア",
    "「愛されたいって願ってる [Cover]（feat.可不）」 - MIMI",
    "「キュートなカノジョ（feat.可不）」 - syudou",
    "「クレイジー・モスキート（feat.可不）」 - 口癖ハマー",
    "「からかい問題（feat.可不 & v flower）」 - こわどり",
    "「ギフト（feat.可不）」 - 内緒のピアス",
    "「ヒミツ（feat.可不）」 - MIMI",
    "「碧空（feat.可不）」 - Chill Hana.",
    "「人と違う（feat.可不）」 - モ葛",
    "「はぐ（feat.可不 & 初音ミク）」 - MIMI",
    "「メスト（feat.可不）」 - かいりきベア",
    "「由々しいあなた（feat.可不）」 - 中瀬ミル",
    "「夜の自販機（feat.可不）」 - にほしか",
    "「タクシィ（feat.可不）」 - Chinozo",
    "「下剋上（feat.可不）」 - Misumi",
    "「負けヒロイン（feat.可不）」 - HASU",
    "「カレイド（feat.可不）」 - 雄之助",
    "「抱きしめて欲しかったなんてね（feat.可不）」 - モ葛",
    "「シロクロアノニマス（feat.可不）」 - 雨良 Amala",
    "「SiGN（feat.可不 & GUMI & 初音ミク & 重音テト）」 - ナナホシ管弦楽団",
    "「南無（feat.可不）」 - 妖師",
    "「空を満たして（feat.可不）」 - フロクロ",
    "「普通.gif（feat.可不）」 - A4。",
    "「メテオラ（feat.可不 & v flower）」 - Chinozo",
    "「天使の翼。（feat.可不 & ゲキヤク）」 - A4。",
    "「ジェラシス（feat.可不）」 - Chinozo",
    "「水葬（feat.可不）」 - Omochi",
    "「マーシャル・マキシマイザー - 花譜（feat.可不）」 - 花譜",
    "「ねこふんじゃった。（feat.可不）」 - A4。",
    "「バースデー（feat.可不）」 - クラム",
    "「まだ貴方のおはようを待っている。（feat.可不 & 星界）」 - 夜未アガリ",
    "「貴方の恋人になりたい [Cover]（feat.可不）」 - tawase",
    "「声色（feat.可不）」 - ornot",
    "「Aye, aye, sir!（feat.可不 & 初音ミク & 重音テト）」 - さたぱんP",
    "「前パッツンLOVE（feat.可不）」 - 可不",
    "「Plazma [TETO with KAFU Cover]（feat.可不 & 重音テト）」 - 可不",
    "「うらやみしい（feat.可不 & Kai & 初音ミク）」 - Loveit Core",
    "「めめしい（feat.可不）」 - すりぃ",
    "「流線形メーデー（feat.可不 & 花譜）」 - 花譜",
    "「ナイティアラート（feat.可不 & 雨衣）」 - DIORAMA",
    "「さまば（feat.可不）」 - 三ケ秋",
    "「消えない温度 [Cover]（feat.可不）」 - MIMI",
    "「エリート（feat.可不）」 - Chinozo",
    "「九段下パンデミック。（feat.可不）」 - A4。",
    "「させて量産（feat.可不）」 - パンクス",
    "「幽霊病（feat.可不）」 - 凪ヤナリ",
    "「前を向かなきゃ（feat.可不）」 - 水野あつ",
    "「アイスクリーム（feat.可不）」 - Guiano",
    "「IRIS OUT [Teto Cover]（feat.可不 & 重音テト）」 - 可不",
    "「怪物 [Cover]（feat.可不）」 - 可不",
    "「ある少女の始末（feat.可不）」 - Swenchy",
    "「Fallen（feat.可不 & 裏命）」 - pupa",
    "「Dull!!（feat.可不）」 - 飽海",
    "「裏世界（feat.可不）」 - niki",
    "「ふわり（feat.可不 & MIMI & 初音ミク）」 - Loveit Core",
    "「ヘテロ（feat.可不 & 歌愛ユキ）」 - 柏木カレキ",
    "「私のせいじゃない（feat.可不 & 歌愛ユキ）」 - 才歌",
    "「いまいち（feat.可不）」 - NAME.O",
    "「縛（feat.可不）」 - でんの子P",
    "「星期零（feat.可不 & 重音テト）」 - s62",
    "「ナイトルール（feat.可不）」 - 煮ル果実",
    "「ねむれないよる（feat.可不）」 - SHIZUKU",
    "「もっとかわいい（feat.可不）」 - 才歌",
    "「さくらさくらさくら（feat.可不）」 - South&",
    "「エメラルド婚式（feat.可不）」 - マユ太",
    "「ノートリアス（feat.可不）」 - てにをは",
    "「アタシ：アップデート（feat.可不）」 - 香椎モイミ",
    "「Mud Princeeeees!!!（feat.可不）」 - A4。",
    "「フォニイ - 花譜（feat.可不）」 - 花譜",
    "「プルメリア。（feat.可不 & #kzn & ci flower & 裏命）」 - A4。",
    "「飾って（feat.可不）」 - 大沼パセリ",
    "「それで充分だよ。（feat.可不）」 - MIMI",
    "「コぇちっちゃくてゴ×ンネ（feat.可不）」 - cosMo@暴走P",
    "「ヒロイン（feat.可不 & なこたんまる）」 - PIKASONIC",
    "「今はいいんだよ。（feat.可不）」 - MIMI",
    "「昏い夜（feat.可不 & 吐息 & v flower）」 - Loveit Core",
    "「サヨナラは言わないでさ（feat.可不）」 - MIMI",
    "「クィホーティ（feat.可不）」 - エイハブ",
    "「愛するように（feat.可不）」 - MIMI",
    "「ありあ（feat.可不）」 - MIMI",
    "「ファントマ（feat.可不）」 - てにをは",
    "「脳裏のマキナ（feat.可不）」 - Folicca",
    "「人マニア [Cover]（feat.可不）」 - tawase",
    "「めめしい - 花譜（feat.可不）」 - 花譜",
    "「ひみつのユーフォー（feat.可不）」 - ナユタン星人",
    "「Need You（feat.可不）」 - EMIRI",
    "「違うよ～＾＾（feat.可不 & 初音ミク）」 - A4。",
    "「シニカルディストピア（feat.可不 & v flower）」 - Fty",
    "「心を刺す言葉だけ（feat.可不 & 初音ミク）」 - MIMI",
    "「閃耀（feat.可不）」 - Feryquitous",
    "「メンがヘラるも好きのうち（feat.可不）」 - Shiero",
    "「きゅうくらりん（feat.可不）」 - いよわ",
    "「ツイッターランド（feat.可不）」 - STEAKA",
    "「ストックホルムオフィス（feat.可不）」 - クラム",
    "「死んでしまいたい夜に（feat.可不）」 - らぴら",
    "「くうになる（feat.可不 & 初音ミク）」 - MIMI",
    "「戻るボタン（feat.可不）」 - なみぐる",
    "「私は、私達は（feat.可不）」 - Guiano",
    "「あかず（feat.可不）」 - かたぎり",
    "「COMEDY。（feat.可不）」 - A4。",
    "「だきしめるまで。（feat.可不）」 - MIMI",
    "「dufrest（feat.可不 & ナースロボ_タイプT）」 - *Natete.",
    "「死んだらさ、（feat.可不）」 - ツナ。",
    "「造花の道、歩めば（feat.可不）」 - South&",
    "「ちょっかい問題（feat.可不 & v flower）」 - こわどり",
    "「可愛くてごめん [Cover]（feat.可不）」 - tawase",
    "「リア（feat.可不）」 - A4。",
    "「Nosy（feat.可不）」 - pupa",
    "「なにやってもうまくいかない（feat.可不）」 - 可不",
    "「大丈夫だよ。（feat.可不）」 - MIMI",
    "「プロポーズ（feat.可不）」 - 内緒のピアス",
    "「絶望。（feat.可不）」 - A4。",
    "「はいジョージ（feat.可不）」 - HorseSea1",
    "「電脳眠眠猫（feat.可不）」 - なみぐる",
    "「女孩和兔子朋友（feat.可不 & 重音テト）」 - 热寂",
    "「ポシェット（feat.可不）」 - MIMI",
    "「抱きしめるだけ。（feat.可不）」 - りび",
    "「可不ちゃんのカレーうどん狂騒曲（feat.可不 & ずんだもん）」 - 南ノ南",
    "「化孵化(cover)（feat.可不/花譜）」 - sasakure.UK",
    "「朝日（feat.可不 & 花譜）」 - カンザキイオリ（黑柿子/神崎伊織/神綺一織）",
    "「メモリー（feat.可不 & 星界 & ナースロボ_タイプT）」 - kyiku",
    "「ちょっとあざとい（feat.可不）」 - 才歌",
    "「可不ェイン（feat.可不）」 - 柊マグネタイト（柊磁鐵）",
    "「死にたいわけじゃなくて（feat.可不）」 - アサノマチ",
    "「水流音楽（feat.可不）」 - MIMI",
    "「アイと勿忘草（feat.可不）」 - South&",
    "「はこにわはゆめのなか（feat.可不）」 - あーる",
    "「ダリアダリア（feat.可不）」 - ねじ式",
    "「バレリーナ.jpeg（feat.可不）」 - A4。",
    "「フォニイ（feat.可不）」 - ツミキ",
    "「Corpse Reviver（feat.可不）」 - Ton Magie",
    "「バースデイ（feat.可不）」 - 内緒のピアス",
    "「ラブリーマインガール（feat.可不 & りむる）」 - litmus*",
    "「Dinner（feat.可不）」 - 雨良 Amala",
    "「裏命ちゃんのフクオカトリップ奇騒曲（feat.可不 & 裏命 & 星界 & 東北きりたん & フリモメン & 松曄りすく）」 - 南ノ南",
    "「このままの心で怖かった（feat.可不 & 重音テト）」 - kyiku",
    "「ジブラ（feat.可不 & 鏡音レン）」 - すりぃ",
    "「愛撫誘発性攻撃行動（feat.可不）」 - Coward Dream",
    "「うゆゆ(；ω；｀)（feat.可不）」 - なみぐる",
    "「六角形のカフカ（feat.可不）」 - STEAKA",
    "「妄想哀歌（feat.可不 & 初音ミク）」 - MIMI",
    "「カノン（feat.可不）」 - 柊マグネタイト（柊磁鐵）",
    "「花となれ（feat.可不 & 攻）」 - 雄之助",
    "「逢瀬の夢（feat.可不）」 - UZURA",
    "「またおいで（feat.可不）」 - South&",
    "「ハナタバ（feat.可不）」 - MIMI",
    "「ドリームイーター（feat.可不 & なこたんまる）」 - PIKASONIC",
    "「感情は成仏しない（feat.可不）」 - IQYU",
    "「フィオーレ（feat.可不 & 初音ミク）」 - MIMI",
    "「GURU (ボカロ盤バージョン)（feat.可不）」 - じん",
    "「嗯嗯。混蛋世界。（feat.可不）」 - 小鷹",
    "「明日が来るのが怖くてさ（feat.可不）」 - さくらかわ",
    "「アイロニカルユートピア（feat.可不 & v flower）」 - Fty",
    "「私のドッペルゲンガー（feat.可不）」 - DIVELA",
    "「あのね（feat.可不）」 - MIMI",
    "「うらら（feat.可不）」 - Gamio",
    "「Devil（feat.可不）」 - てにをは",
    "「あいされたい（feat.可不 & 重音テト & ナースロボ_タイプT）」 - 雨良 Amala",
    "「特別救済委員会（feat.可不 & 初音ミク & GUMI & IA & v4 flower & 重音テト）」 - サツキ",
    "「愛し愛（feat.可不 & 初音ミク）」 - MIMI",
    "「無理に笑わなくて良いよ（feat.可不）」 - 水野あつ",
    "「いっせーのーで（feat.可不）」 - MIMI",
    "「オマジナイ（feat.可不）」 - MIMI",
    "「社会距離（feat.可不）」 - 40mP",
    "「反実夢想（feat.可不）」 - なみぐる",
    "「花火が落ちる前に（feat.可不）」 - 糖分",
    "「ソルティメロウ（feat.可不）」 - MIMI",
    "「死んでしまったんだ（feat.可不 & 結月ゆかり）」 - 椎乃味醂",
    "「I♡ [可不ver]（feat.可不）」 - 名前は未だ無いです。",
    "「キャットラビング - 花譜（feat.可不）」 - 花譜",
    "「撫でんな（feat.可不）」 - 柊マグネタイト（柊磁鐵）",
    "「うつろうタイム（feat.可不）」 - Nanashi_Zero",
    "「不埒な喝采（feat.可不）」 - 花譜",
    "「ハナビラ（feat.可不 & 星界）」 - SHIZUKU",
    "「息をするだけ（feat.可不）」 - MIMI",
    "「縦読み問題（feat.可不 & v flower）」 - こわどり",
    "「レトロポリス（feat.可不）」 - R Sound Design",
    "「ようやく君が死んだんだ。（feat.可不）」 - Coward Dream",
    "「君が僕を嗤う日（feat.可不）」 - 中瀬ミル",
    "「チーズ（feat.可不）」 - Chinozo",
    "「鳥（feat.可不）」 - Guiano",
    "「アイソトープ（feat.可不）」 - r-906",
    "「CH4NGE (TeddyLoid Remix)（feat.可不）」 - Giga",
    "「どっかいこうよ（feat.可不 & 初音ミク）」 - reinou",
    "「生きる（feat.可不）」 - 水野あつ",
    "「星界ちゃんと可不ちゃんのおつかい合騒曲（feat.可不 & 星界）」 - 南ノ南",
    "「弱気なday（feat.可不）」 - HorseSea1",
    "「永遠にさよなら（feat.可不）」 - 哲作",
    "「ただ病名が欲しかった（feat.可不 & 星界 & カゼヒキ）」 - kyiku",
    "「Stairway in the void（feat.可不）」 - 可不",
    "「マシュマロチューイングガム（feat.可不）」 - 海ジャム",
    "「Rabbit!?。（feat.可不 & #kzn & v flower）」 - A4。",
    "「キャットラビング（feat.可不）」 - 香椎モイミ",
    "「シルバーツインズ（feat.可不 & v flower）」 - 蜂屋ななし",
    "「超スーパーウルトラホット（feat.可不 & 初音ミク & 音街ウナ）」 - メドミア",
    "「不埒な喝采（feat.可不）」 - ポリスピカデリー",
]


BIGRAM_SIZE = 2


def sort_by_common_bigrams(
    text: str, string_list: list[str]
) -> list[tuple[str, float]]:
    def _jaccard_bigram(a: str, b: str) -> float:
        a_bg = {a[i : i + BIGRAM_SIZE] for i in range(len(a) - 1)}
        b_bg = {b[i : i + BIGRAM_SIZE] for i in range(len(b) - 1)}
        union = len(a_bg | b_bg)
        return len(a_bg & b_bg) / union if union else 0.0

    scored = []
    for s in string_list:
        score = (
            SequenceMatcher(None, text, s).ratio() * 0.5
            + _jaccard_bigram(text, s) * 0.5
        )
        scored.append((s, score))
    return scored


@tool(
    description="根据关键词搜索歌曲，请输入关键词，多个关键词用空格或逗号分隔。与歌曲相关的问题，必须要使用此工具。"
)
def search_song(keywords: str) -> str:
    t_text, s_text = (
        opencc.OpenCC("s2t").convert(keywords),
        opencc.OpenCC("t2s").convert(keywords),
    )
    t_songs = sort_by_common_bigrams(t_text, SONGS_LIST)
    s_songs = sort_by_common_bigrams(s_text, SONGS_LIST)
    all_songs = t_songs + s_songs
    if not all_songs:
        return "Can't find any songs."
    best_score: dict[str, float] = {}
    for song, score in all_songs:
        if song not in best_score or score > best_score[song]:
            best_score[song] = score
    sorted_songs = list(best_score.items())
    sorted_songs.sort(key=lambda x: x[1], reverse=True)
    top_songs = [song for song, _ in sorted_songs[:5]]
    return KAFU_SONGS_PROMPT + "\n".join(f"- {song}" for song in top_songs)


def apply(context: PluginContext, _config: Any) -> None:
    context.tool(search_song, ToolScope.OPEN)


plugin = PluginDefinition(name="search_song", apply=apply)


if __name__ == "__main__":
    print(search_song.invoke({"keywords": "可不 星界"}))
