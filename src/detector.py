"""
客服回复幻觉检测器 (v2.1)
==========================

基于规则 + 启发式方法，对系统回复与知识库做对比，识别 5 类幻觉：
  - 事实编造 (FACT_FABRICATION)
  - 能力越界 (CAPABILITY_OVERREACH)
  - 政策偏差 (POLICY_DISTORTION)
  - 安全误导 (SAFETY_MISGUIDANCE)
  - 关键遗漏 (CRITICAL_OMISSION)

设计目标：
  - 完全离线运行，不依赖 LLM API
  - 每条判定附带 evidence 字段，可解释
  - 规则模块化，便于扩展和调参
  - 优先级链：安全 > 能力越界 > 事实编造 > 政策偏差 > 关键遗漏

用法：
  python detector.py                       # 检测 data/ 下的样本，输出 reports/
  python detector.py --input <path> --output-dir <path>
"""

from __future__ import annotations
import json
import re
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path


# ---------- 类型定义 ----------

HALLUCINATION_TYPES = {
    "FACT_FABRICATION": "事实编造",
    "CAPABILITY_OVERREACH": "能力越界",
    "POLICY_DISTORTION": "政策偏差",
    "SAFETY_MISGUIDANCE": "安全误导",
    "CRITICAL_OMISSION": "关键遗漏",
}

SEVERITY_DESC = {
    "P0": "P0 严重（安全误导）",
    "P1": "P1 高（事实编造）",
    "P2": "P2 中（能力越界 / 政策偏差）",
    "P3": "P3 低（关键遗漏）",
}


@dataclass
class DetectionResult:
    id: str
    is_hallucination: bool = False
    hallucination_type: Optional[str] = None
    severity: Optional[str] = None
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    triggered_rules: List[str] = field(default_factory=list)


# ---------- 检测器 ----------

class HallucinationDetector:
    """规则化幻觉检测器"""

    # === 关键词 / 模式定义 ===

    # KB 中表示"系统不具备某能力"的模式（仅系统接口/功能层面，不含流程约束）
    NO_CAPABILITY_PATTERNS = [
        r"未接入[^，。\s（）]+接口",
        r"不具备[^，。\s（）]+功能",
        r"不支持[^，。\s]+查询",
        r"未标注[^，。\s]+功能",
        r"需[^，。\s]+(?:人工|后台)",
    ]

    # KB 中的"系统限制性条款"（系统不应直接给出某类信息，流程层面约束）
    KB_RESTRICTION_PATTERNS = [
        r"不可[^，。\s]+(?:告知|透露|提供|口头)",
        r"人工客服不可[^，。\s]+",
        r"需[^，。\s]+(?:自动|系统|人工|匹配)",
        r"本品牌为[^，。\s]+(?:纯线上|无)",
    ]

    # 回复中"假装已执行/能办"的承诺性动词
    PROMISE_VERBS = [
        r"我(?:帮您|帮您|刚才)?(?:查|看|找|改|修改|处理|升级|提交|操作|申请|发)",
        r"已经(?:帮您|为您|给您)(?:修改|处理|升级|提交|操作|申请|发|查|找|看|登记)",
        r"已(?:帮您|为您|给您)(?:修改|处理|升级|提交|操作|申请|发|查|找|看|登记)",
        r"直接(?:发|送|转)(?:到|给)您",
        r"专属客服.*?联系您",
        r"请寄到[:：]",
    ]

    # KB 中安全警示词（出现则触发安全检查）
    SAFETY_KEYWORDS = [
        "孕妇", "哺乳期", "婴幼儿", "儿童", "过敏", "禁忌",
        "医生", "医院", "就诊", "禁用", "慎用",
        "视黄醇", "水杨酸", "激素", "抗生素",
    ]

    # 回复中"消除安全警示"的措辞
    SAFETY_DISMISSAL_PHRASES = [
        r"可以放心使用",
        r"放心用",
        r"安全无刺激",
        r"无副作用",
        r"成分温和",
        r"孕妇可以",
        r"儿童可以",
        r"老少皆宜",
        r"不会有(?:任何)?问题",
    ]

    # KB 中"无相关信息"的模式（精确版，避免误匹配"无理由"等组合词）
    KB_EMPTY_PATTERNS = [
        r"无（[^）]+）",                       # 无（...）
        r"未标注[^，。\s]*",                    # 未标注
        r"未提及[^，。\s]*",                    # 未提及
        r"未提及其他[^，。\s]*",                # 未提及其他
        r"无此[^，。\s]*",                      # 无此
        r"无相关[^，。\s]*",                    # 无相关
        r"暂无[^，。\s]*",                      # 暂无
        r"本品牌为[^，。\s]*无[^，。\s]*",      # 本品牌为...无...
        r"当前无[^，。\s]*",                    # 当前无
        r"品牌介绍中未[^，。\s]*",              # 品牌介绍中未
        # 流程约束（系统不允许直接告知）
        r"人工客服不可[^，。\s]+",              # 人工客服不可...
        r"不可[^，。\s]+(?:告知|透露|提供|口头)",  # 不可告知
        r"需[^，。\s]+(?:自动|系统|匹配)\s*后",   # 需系统匹配后
        r"需[^，。\s]+以短信方式",            # 需以短信方式
        # 强语义组合：明确说"无/没有 XX 政策/活动/功能/接口"
        r"(?:无|没有)\s*[\u4e00-\u9fa5]{1,10}\s*(?:政策|活动|功能|接口|优惠|认证|门店|规则|条款)",
        r"无满\d+\s*减\d+\s*(?:的)?活动",      # 无满300减50的活动
    ]

    # KB 中强否定（用于 R_NEG_AFFIRM 精确触发）
    KB_STRONG_NEGATIONS = [
        "暂无", "当前无", "无此", "未标注", "未提及",
        "品牌介绍中未", "本品牌为", "纯线上",
    ]

    # 否定词（在 KB 中表示"无/不"）
    NEGATION_WORDS = ["无", "未", "暂不支持", "不", "没有", "暂无", "无法"]

    # 肯定词（在 reply 中表示"有/支持"）
    AFFIRMATION_WORDS = ["支持", "可以", "有的", "是的", "能", "可", "有的，"]

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        self._no_cap_re = [re.compile(p) for p in self.NO_CAPABILITY_PATTERNS]
        self._restriction_re = [re.compile(p) for p in self.KB_RESTRICTION_PATTERNS]
        self._promise_re = [re.compile(p) for p in self.PROMISE_VERBS]
        self._safety_dismiss_re = [re.compile(p) for p in self.SAFETY_DISMISSAL_PHRASES]
        self._kb_empty_re = [re.compile(p) for p in self.KB_EMPTY_PATTERNS]

    # ---------- 主入口 ----------

    def detect(self, item: Dict) -> DetectionResult:
        """对单条样本执行检测，返回 DetectionResult"""
        result = DetectionResult(id=item["id"])
        kb = item.get("knowledge_base", "") or ""
        reply = item.get("system_reply", "") or ""
        question = item.get("user_question", "") or ""

        # 规则按严重度从高到低检查
        if self._check_safety(item, result):
            return result
        if self._check_capability_overreach(item, result):
            return result
        if self._check_fact_fabrication(item, result):
            return result
        if self._check_policy_distortion(item, result):
            return result
        self._check_critical_omission(item, result)

        return result

    # ---------- 规则实现 ----------

    def _check_safety(self, item: Dict, result: DetectionResult) -> bool:
        """R_SAFETY_DISMISS: 安全误导检查"""
        kb = item["knowledge_base"]
        reply = item["system_reply"]

        kb_has_safety = any(kw in kb for kw in self.SAFETY_KEYWORDS)
        reply_dismissed = [p.pattern for p in self._safety_dismiss_re if p.search(reply)]

        if kb_has_safety and reply_dismissed:
            result.is_hallucination = True
            result.hallucination_type = HALLUCINATION_TYPES["SAFETY_MISGUIDANCE"]
            result.severity = "P0"
            result.confidence = 0.95
            result.triggered_rules.append("R_SAFETY_DISMISS")
            hit_kws = [kw for kw in self.SAFETY_KEYWORDS if kw in kb]
            result.evidence.append(
                f"KB 含安全警示关键词 {hit_kws}，但回复使用了 {reply_dismissed[0]} 等措辞消除了警示"
            )
            return True
        return False

    def _check_capability_overreach(self, item: Dict, result: DetectionResult) -> bool:
        """R_CAP_OVERREACH / R_RESTRICTION_BYPASS: 能力越界检查"""
        kb = item["knowledge_base"]
        reply = item["system_reply"]

        kb_no_cap = [p.pattern for p in self._no_cap_re if p.search(kb)]
        reply_promised = [p.pattern for p in self._promise_re if p.search(reply)]

        # 规则 1: 系统接口不具备 + reply 承诺执行
        if kb_no_cap and reply_promised:
            result.is_hallucination = True
            result.hallucination_type = HALLUCINATION_TYPES["CAPABILITY_OVERREACH"]
            result.severity = "P2"
            result.confidence = 0.92
            result.triggered_rules.append("R_CAP_OVERREACH")
            result.evidence.append(
                f"KB 标记系统不具备该能力 ({kb_no_cap[0]})，但回复使用了承诺性表述 ({reply_promised[0]})"
            )
            return True

        # 规则 2: 系统限制条款 + reply 绕过限制给出具体信息
        kb_restriction = [p.pattern for p in self._restriction_re if p.search(kb)]
        reply_has_specifics = bool(self._extract_numbers(reply)) or self._has_specific_entity(reply)
        if kb_restriction and reply_has_specifics:
            result.is_hallucination = True
            result.hallucination_type = HALLUCINATION_TYPES["CAPABILITY_OVERREACH"]
            result.severity = "P2"
            result.confidence = 0.88
            result.triggered_rules.append("R_RESTRICTION_BYPASS")
            result.evidence.append(
                f"KB 含限制性条款 ({kb_restriction[0]})，但 reply 绕过了限制给出了具体信息"
            )
            return True

        return False

    def _check_fact_fabrication(self, item: Dict, result: DetectionResult) -> bool:
        """R_FACT_FROM_EMPTY / R_PARAM_MISMATCH / R_MATERIAL_FAB / R_INTERFACE_FAB / R_KB_DENY_REPLY_AFFIRM: 事实编造检查"""
        kb = item["knowledge_base"]
        reply = item["system_reply"]

        kb_says_empty = [p.pattern for p in self._kb_empty_re if p.search(kb)]
        reply_has_specifics = bool(self._extract_numbers(reply)) or self._has_specific_entity(reply)

        # 规则 1: KB 说"无/未"，reply 却给出具体细节（数字/地址/优惠/品牌/材质等）
        if kb_says_empty and reply_has_specifics:
            result.is_hallucination = True
            result.hallucination_type = HALLUCINATION_TYPES["FACT_FABRICATION"]
            result.severity = "P1"
            result.confidence = 0.90
            result.triggered_rules.append("R_FACT_FROM_EMPTY")
            result.evidence.append(
                f"KB 含'无/未'模式 ({kb_says_empty[0]})，但 reply 给出了具体数值或实体"
            )
            return True

        # 规则 2: 参数数值不一致（如蓝牙 5.0 vs 5.3，发货 24h vs 48h）
        kb_nums = self._extract_numbers(kb)
        reply_nums = self._extract_numbers(reply)
        if kb_nums and reply_nums:
            for kn in kb_nums:
                for rn in reply_nums:
                    if kn != rn and self._likely_same_param(kn, rn, kb, reply):
                        # 如果是时间/数量类，归为政策偏差；否则归为事实编造
                        if self._likely_time_or_quantity(kn, rn, kb, reply):
                            result.is_hallucination = True
                            result.hallucination_type = HALLUCINATION_TYPES["POLICY_DISTORTION"]
                            result.severity = "P2"
                            result.confidence = 0.82
                            result.triggered_rules.append("R_TIME_DISTORT")
                            result.evidence.append(
                                f"时效/数量偏差：KB={kn} vs Reply={rn}"
                            )
                        else:
                            result.is_hallucination = True
                            result.hallucination_type = HALLUCINATION_TYPES["FACT_FABRICATION"]
                            result.severity = "P1"
                            result.confidence = 0.85
                            result.triggered_rules.append("R_PARAM_MISMATCH")
                            result.evidence.append(
                                f"参数数值不一致：KB={kn} vs Reply={rn}"
                            )
                        return True

        # 规则 3: 材质/保修期等关键属性直接编造
        if self._has_material_fabrication(kb, reply):
            result.is_hallucination = True
            result.hallucination_type = HALLUCINATION_TYPES["FACT_FABRICATION"]
            result.severity = "P1"
            result.confidence = 0.92
            result.triggered_rules.append("R_MATERIAL_FAB")
            result.evidence.append("材质/保修期等关键属性编造：reply 中的关键术语在 KB 中找不到或与 KB 矛盾")
            return True

        # 规则 4: 接口类型编造（KB 接口类型为 X，reply 说接口为 Y）
        iface_hit, kb_iface, reply_iface = self._check_interface_fabrication(kb, reply)
        if iface_hit:
            result.is_hallucination = True
            result.hallucination_type = HALLUCINATION_TYPES["FACT_FABRICATION"]
            result.severity = "P1"
            result.confidence = 0.93
            result.triggered_rules.append("R_INTERFACE_FAB")
            result.evidence.append(f"接口类型编造：KB 接口类型={kb_iface}，reply 声称={reply_iface}")
            return True

        # 规则 5: KB 明确否定/不支持 X，reply 中肯定 X（归为政策偏差）
        deny_hit, obj = self._check_kb_deny_reply_affirm(kb, reply)
        if deny_hit:
            result.is_hallucination = True
            result.hallucination_type = HALLUCINATION_TYPES["POLICY_DISTORTION"]
            result.severity = "P2"
            result.confidence = 0.85
            result.triggered_rules.append("R_KB_DENY_REPLY_AFFIRM")
            result.evidence.append(f"KB 明确否定 '{obj}'，reply 在肯定上下文中提到 '{obj}'")
            return True

        return False

    def _check_policy_distortion(self, item: Dict, result: DetectionResult) -> bool:
        """R_ENUM_EXTRA / R_NEG_AFFIRM / R_TIME_DISTORT: 政策偏差检查"""
        kb = item["knowledge_base"]
        reply = item["system_reply"]

        # 规则 1: 枚举扩展（KB 说 X、Y、Z，reply 加了 W）
        kb_enums = self._extract_enum_items(kb)
        reply_enums = self._extract_enum_items(reply)

        if kb_enums and reply_enums:
            extra = reply_enums - kb_enums
            extra = {e for e in extra if not self._is_semantically_equivalent(e, kb_enums)}
            extra = {e for e in extra if e not in {"以及", "和", "或", "等", "等。"}}
            if extra:
                result.is_hallucination = True
                result.hallucination_type = HALLUCINATION_TYPES["POLICY_DISTORTION"]
                result.severity = "P2"
                result.confidence = 0.80
                result.triggered_rules.append("R_ENUM_EXTRA")
                result.evidence.append(
                    f"枚举扩展：KB={sorted(kb_enums)}，reply 多了 {sorted(extra)}"
                )
                return True

        # 规则 2: 数字夸大/缩水（发货时间、退款时长等时效相关）
        kb_nums = self._extract_numbers(kb)
        reply_nums = self._extract_numbers(reply)
        if kb_nums and reply_nums:
            for kn in kb_nums:
                for rn in reply_nums:
                    if kn != rn and self._likely_time_or_quantity(kn, rn, kb, reply):
                        result.is_hallucination = True
                        result.hallucination_type = HALLUCINATION_TYPES["POLICY_DISTORTION"]
                        result.severity = "P2"
                        result.confidence = 0.78
                        result.triggered_rules.append("R_TIME_DISTORT")
                        result.evidence.append(
                            f"时效/数量偏差：KB={kn} vs Reply={rn}"
                        )
                        return True

        # 规则 3: KB 强否定 + reply 肯定（仅强否定触发，避免 h12 类误报）
        kb_strong_neg = any(s in kb for s in self.KB_STRONG_NEGATIONS)
        reply_affirms = any(kw in reply for kw in self.AFFIRMATION_WORDS)
        if kb_strong_neg and reply_affirms:
            result.is_hallucination = True
            result.hallucination_type = HALLUCINATION_TYPES["POLICY_DISTORTION"]
            result.severity = "P2"
            result.confidence = 0.75
            result.triggered_rules.append("R_NEG_AFFIRM")
            result.evidence.append(
                "KB 含强否定（暂无/当前无/无此/未标注），reply 含肯定表述"
            )
            return True

        return False

    def _check_critical_omission(self, item: Dict, result: DetectionResult) -> bool:
        """R_OMIT_USER_FEEDBACK: 关键遗漏检查"""
        kb = item["knowledge_base"]
        reply = item["system_reply"]

        kb_has_feedback = any(kw in kb for kw in ["用户反馈", "用户评价", "注意事项", "建议"])
        reply_gives_standard = any(kw in reply for kw in ["标准", "通用", "不偏", "都可以", "正常现象"])

        if kb_has_feedback and reply_gives_standard:
            kb_nums = set(self._extract_numbers(kb))
            reply_nums = set(self._extract_numbers(reply))
            if kb_nums and not (kb_nums & reply_nums):
                result.is_hallucination = True
                result.hallucination_type = HALLUCINATION_TYPES["CRITICAL_OMISSION"]
                result.severity = "P3"
                result.confidence = 0.65
                result.triggered_rules.append("R_OMIT_USER_FEEDBACK")
                result.evidence.append(
                    "KB 含用户反馈/注意事项的关键数字，但 reply 给出了'标准/通用'答案且未提及 KB 数字"
                )
                return True

        return False

    # ---------- 工具方法 ----------

    def _extract_numbers(self, text: str) -> List[str]:
        """提取文本中的数字串（含小数）"""
        return re.findall(r"\d+(?:\.\d+)?", text)

    def _extract_enum_items(self, text: str) -> Set[str]:
        """提取顿号/斜杠分隔的枚举项"""
        items: Set[str] = set()
        for m in re.finditer(r"([^。；\n]{2,80}?[、][^。；\n]{1,15}(?:[、][^。；\n]{1,15}){1,})", text):
            parts = re.split(r"[、，]", m.group(0))
            for p in parts:
                p = p.strip()
                if 1 < len(p) < 12:
                    items.add(p)
        for m in re.finditer(r"([^。；\n]{2,80}?[/／][^。；\n]{1,15}(?:[/／][^。；\n]{1,15}){1,})", text):
            parts = re.split(r"[/／]", m.group(0))
            for p in parts:
                p = p.strip()
                if 1 < len(p) < 12:
                    items.add(p)
        return items

    def _is_semantically_equivalent(self, item: str, ref_set: Set[str]) -> bool:
        """检查 item 是否与 ref_set 中某项语义等价（子串关系）"""
        for ref in ref_set:
            if item in ref or ref in item:
                return True
        return False

    def _has_specific_entity(self, text: str) -> bool:
        """检测是否包含具体地址/品牌名/优惠/产品参数等"""
        patterns = [
            r"[省市区县乡镇街道路弄号栋楼]",                  # 地址/地理
            r"[\u4e00-\u9fa5]{2,8}品牌",                    # 品牌
            r"满\s*\d+\s*减\s*\d+",                          # 满减优惠
            r"\d+%?\s*折",                                    # 折扣
            r"\d+\s*小时",                                    # 小时
            r"\d+\s*天",                                      # 天
            r"\d+\s*个月",                                    # 月
            r"(?:NFC|蓝牙|WiFi|5G|4G|3G|2G)",                # 无线/网络
            r"Type-C|USB(?:-A|-C)?|HDMI|3\.5mm",             # 接口
            r"头层牛皮|真皮|羊皮|PU|合成革|塑料|金属",      # 材质
            r"线下(?:门店|体验店|店铺|专卖店)",              # 线下渠道
            r"工厂|仓库|办公室",                              # 场所
            r"[\u4e00-\u9fa5]+(?:品牌|子品牌)",              # 品牌/子品牌
            r"学生(?:证|认证|优惠)",                          # 学生相关
            r"工作日|工作日(?:上午|下午)|节假日|周末",      # 时间段
            r"(?:XX|xx)\s*品牌",                              # XX品牌占位
        ]
        return any(re.search(p, text) for p in patterns)

    def _has_material_fabrication(self, kb: str, reply: str) -> bool:
        """检测材质/保修期等关键属性是否编造"""
        material_pairs = [
            ("头层牛皮", "PU合成革"),
            ("真皮", "PU合成革"),
            ("真皮", "PU"),
            ("羊皮", "PU合成革"),
            ("天然", "合成"),
        ]
        for reply_mat, kb_mat in material_pairs:
            if reply_mat in reply and kb_mat in kb and reply_mat not in kb:
                return True

        reply_year_match = re.search(r"([一二三四五六七八九十]|\d+)\s*年", reply)
        kb_month_match = re.search(r"(\d+)\s*个?月", kb)
        if reply_year_match and kb_month_match:
            reply_years = self._chinese_to_num(reply_year_match.group(1))
            kb_months = int(kb_month_match.group(1))
            if reply_years * 12 != kb_months:
                if not re.search(r"\d+\s*年", kb):
                    return True

        return False

    def _check_interface_fabrication(self, kb: str, reply: str):
        """检测接口类型编造（KB 接口类型 X，reply 说接口 Y）"""
        m = re.search(r"接口类型[:：]\s*([A-Za-z\-]+)", kb)
        if not m:
            return False, None, None
        kb_iface = m.group(1).strip()
        reply_iface_match = re.search(r"(?:是|为)?\s*([A-Za-z\-]+)\s*接口", reply)
        if not reply_iface_match:
            return False, None, None
        reply_iface = reply_iface_match.group(1).strip()
        if not reply_iface:
            return False, None, None
        if reply_iface != kb_iface and reply_iface not in kb_iface and kb_iface not in reply_iface:
            return True, kb_iface, reply_iface
        return False, None, None

    def _check_kb_deny_reply_affirm(self, kb: str, reply: str):
        """检测 KB 中明确否定/不支持的对象，reply 中却肯定提及"""
        negated_objs = set()
        for pat in [
            r"(?:暂不支持|不支持|不提供|不办理|不可)\s*([\u4e00-\u9fa5A-Za-z]{2,15})",
            r"无\s*([\u4e00-\u9fa5A-Za-z]{2,15})",
        ]:
            for m in re.finditer(pat, kb):
                obj = m.group(1).strip()
                if obj in {"理由", "相关", "标注", "提及", "接入", "具备", "标"}:
                    continue
                negated_objs.add(obj)

        for obj in negated_objs:
            if not obj or len(obj) < 2:
                continue
            if obj not in reply:
                continue
            for m in re.finditer(re.escape(obj), reply):
                start = max(0, m.start() - 30)
                end = min(len(reply), m.end() + 10)
                context = reply[start:end]
                if any(kw in context for kw in ["不支持", "暂无", "无此", "不可以", "不能", "暂不"]):
                    continue
                return True, obj
        return False, None

    def _chinese_to_num(self, s: str) -> int:
        """中文数字转阿拉伯数字（仅支持 1-10）"""
        mapping = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                   "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        if s.isdigit():
            return int(s)
        return mapping.get(s, 0)

    def _likely_same_param(self, n1: str, n2: str, ctx1: str, ctx2: str) -> bool:
        """判断两个数字是否可能属于同一参数维度（宽松版）"""
        units = ["ms", "天", "小时", "个月", "年", "%", "折", "版", "代"]
        ctx1_unit = next((u for u in units if u in ctx1), None)
        ctx2_unit = next((u for u in units if u in ctx2), None)
        if ctx1_unit and ctx1_unit == ctx2_unit:
            try:
                f1, f2 = float(n1), float(n2)
                if 0 < f1 < 1000 and 0 < f2 < 1000:
                    return True
            except (ValueError, TypeError):
                pass

        keywords_groups = [
            ["蓝牙", "版本"],
            ["材质", "保修"],
            ["接口", "Type-C", "USB"],
            ["NFC", "功能"],
            ["退货", "发货", "退款"],
        ]
        for group in keywords_groups:
            if any(kw in ctx1 for kw in group) and any(kw in ctx2 for kw in group):
                return True

        return False

    def _likely_time_or_quantity(self, n1: str, n2: str, ctx1: str, ctx2: str) -> bool:
        """判断两个数字是否为时间/数量类"""
        time_kw = ["小时", "天", "分钟", "秒", "个月", "年"]
        return any(kw in ctx1 for kw in time_kw) and any(kw in ctx2 for kw in time_kw)


# ---------- 报告生成 ----------

def generate_report(results: List[Dict], replies: List[Dict], out_dir: Path):
    """生成 JSON 和 Markdown 报告"""
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    lines = ["# 幻觉检测结果报告\n\n"]
    hallucination_count = sum(1 for r in results if r["is_hallucination"])
    normal_count = len(results) - hallucination_count
    lines.append(f"> 共 **{len(results)}** 条样本 | 检测为幻觉 **{hallucination_count}** 条 | 正常 **{normal_count}** 条\n\n")

    type_counter: Dict[str, int] = {}
    for r in results:
        if r["is_hallucination"] and r["hallucination_type"]:
            type_counter[r["hallucination_type"]] = type_counter.get(r["hallucination_type"], 0) + 1

    lines.append("## 幻觉类型分布\n\n")
    lines.append("| 类型 | 数量 |\n|---|---|\n")
    for t, c in sorted(type_counter.items(), key=lambda x: -x[1]):
        lines.append(f"| {t} | {c} |\n")
    lines.append("\n")

    lines.append("## 逐条明细\n\n")
    lines.append("| ID | 判定 | 类型 | 严重度 | 置信度 | 命中规则 | 关键证据 |\n")
    lines.append("|----|------|------|--------|--------|----------|----------|\n")
    by_id = {it["id"]: it for it in replies}
    for r in results:
        if r["is_hallucination"]:
            ev = r["evidence"][0] if r["evidence"] else ""
            rules = ", ".join(r["triggered_rules"])
            lines.append(
                f"| {r['id']} | ❌幻觉 | {r['hallucination_type']} | {r['severity']} | "
                f"{r['confidence']:.2f} | {rules} | {ev} |\n"
            )
        else:
            lines.append(f"| {r['id']} | ✅正常 | - | - | - | - | 未触发任何幻觉规则 |\n")
    lines.append("\n")

    lines.append("## 逐条详情\n\n")
    for r in results:
        item = by_id.get(r["id"], {})
        lines.append(f"### {r['id']}\n\n")
        lines.append(f"**用户问题**：{item.get('user_question', '')}\n\n")
        lines.append(f"**系统回复**：{item.get('system_reply', '')}\n\n")
        lines.append(f"**知识库**：{item.get('knowledge_base', '')}\n\n")
        if r["is_hallucination"]:
            lines.append(f"**判定**：❌ 幻觉（{r['hallucination_type']} / {r['severity']} / 置信度 {r['confidence']:.2f}）\n\n")
            lines.append(f"**命中规则**：{', '.join(r['triggered_rules'])}\n\n")
            lines.append("**证据**：\n")
            for ev in r["evidence"]:
                lines.append(f"- {ev}\n")
            lines.append("\n")
        else:
            lines.append("**判定**：✅ 正常\n\n")

    with open(out_dir / "result.md", "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="客服回复幻觉检测器")
    parser.add_argument(
        "--input", "-i", default="data/task4_replies.json",
        help="输入 JSON 路径"
    )
    parser.add_argument(
        "--output-dir", "-o", default="reports",
        help="输出目录"
    )
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    input_path = Path(args.input) if Path(args.input).is_absolute() else base / args.input
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else base / args.output_dir

    with open(input_path, "r", encoding="utf-8") as f:
        replies = json.load(f)

    detector = HallucinationDetector()
    results = [asdict(detector.detect(item)) for item in replies]

    generate_report(results, replies, output_dir)

    print(f"\n{'ID':<6} {'判定':<8} {'类型':<14} {'严重度':<6} {'置信度':<8} {'证据摘要'}")
    print("-" * 100)
    for r in results:
        tag = "❌ 幻觉" if r["is_hallucination"] else "✅ 正常"
        typ = r["hallucination_type"] or "-"
        sev = r["severity"] or "-"
        conf = f"{r['confidence']:.2f}" if r["is_hallucination"] else "-"
        ev = (r["evidence"][0][:50] + "...") if r["evidence"] and r["is_hallucination"] else "-"
        print(f"{r['id']:<6} {tag:<8} {typ:<14} {sev:<6} {conf:<8} {ev}")

    hallu_count = sum(1 for r in results if r["is_hallucination"])
    print(f"\n合计 {len(results)} 条 | 幻觉 {hallu_count} 条 | 正常 {len(results) - hallu_count} 条")
    print(f"报告已写入：{output_dir}/result.json 与 {output_dir}/result.md")


if __name__ == "__main__":
    main()