import os
import json
import re
from typing import List, Tuple, Dict, Any

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, AutoModel


def _dedupe_preserve_order(items):
    seen = set()
    deduped = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _clean_item(item: Any) -> str:
    text = str(item).strip().strip("'\"`[]()（）")
    text = re.sub(r"^\s*[\d一二三四五六七八九十]+[.、)\uff09]\s*", "", text)
    return text.strip()


def _parse_items(response: str, allowed_values: List[str] | None = None) -> List[str]:
    """Parse LLM list output from JSON, Chinese punctuation, or newline text."""
    candidates = []
    text = response.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            candidates = [_clean_item(item) for item in parsed]
        elif isinstance(parsed, str):
            candidates = [_clean_item(parsed)]
    except json.JSONDecodeError:
        stripped = text.replace("[", "").replace("]", "")
        candidates = [_clean_item(item) for item in re.split(r"[,，、;；\n]+", stripped)]

    candidates = [item for item in candidates if item]
    if allowed_values is None:
        return _dedupe_preserve_order(candidates)

    matched = []
    for candidate in candidates:
        if candidate in allowed_values:
            matched.append(candidate)
            continue
        for value in allowed_values:
            if value in candidate:
                matched.append(value)
    return _dedupe_preserve_order(matched)


def _normalize_sentiment(response: str) -> str:
    alias_map = {
        "正面": "正面",
        "积极": "正面",
        "中立": "中立",
        "中性": "中立",
        "负面": "负面",
        "消极": "负面",
    }
    for alias, normalized in alias_map.items():
        if alias in response:
            return normalized
    return "中立"


FINANCE_EVENT_HIERARCHY = {
    '财务': ['利润公布', '利润预告', '其他财务动态'],
    '股东': ['股东增减持', '解除质押', '股东质押', '其他股东事件'],
    '股票': ['股价变动', '股票状态', '限售股解禁', '股票回购', '股权激励&员工持股计划',
            '限制股解禁', '股票分红', '其他股票事件'],
    '管理': ['董监高动态', '员工动态'],
    '合规信用': ['监管问询', '公司涉诉', '立案调查', '行政处罚', '澄清公告',
               '法律事务', '评级变动', '其他违规事件'],
    '经营': ['项目中标', '其他经营事件', '开展合作', '新公司成立', '销量、份额变动',
           '知识产权', '技术质控、资质变动', '政府补贴', '机构调研', '产能产量变动',
           '项目动态', '产品动态'],
    '融资投资': ['资金流动', '投资事件', '融资融券', '公司上市', '并购重组',
               '股票增发', '其他融资事件']
}
SENTIMENTS = ['正面', '中立', '负面']


class FinancialEventReasoningAnalyzer:
    """基于推理链的金融事件细粒度情感分析器"""

    def __init__(self, model_name: str):
        """初始化模型和tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()

        self.finance_event_hierarchy = {
            label: list(fine_labels)
            for label, fine_labels in FINANCE_EVENT_HIERARCHY.items()
        }
        self.sentiments = list(SENTIMENTS)
        self.company_industry_map = {}  # 公司-行业映射，可扩展

    def step1_extract_companies(self, news_text: str) -> List[str]:
        """Step 1: 识别新闻中涉及的全部公司"""
        prompt = f"""
        假设你是一个金融领域的实体识别模型，请从以下金融新闻中识别并提取所有涉及的公司名称。
        注意：可能存在多家公司，请全部列出。
        只返回 JSON 数组格式的公司名称列表，不要回答多余的话。

        金融新闻：{news_text}
        """
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)

        # 解析公司名称列表
        try:
            companies = _parse_items(response)
            if companies:
                return companies
        except:
            pass

        # 备选：直接按逗号分割
        companies = _parse_items(response)
        return companies if companies else [self._fallback_extract_company(news_text)]

    def _fallback_extract_company(self, news_text: str) -> str:
        """备选方案：提取单个公司名称"""
        prompt = f"""
        从以下金融新闻中识别出主要的公司名称，只回答公司名称，不要多余的话。
        金融新闻：{news_text}
        """
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)
        return response.strip()

    def step2_extract_coarse_events(self, news_text: str, company: str) -> List[str]:
        """Step 2: 对公司提取所有一级事件"""
        coarse_events = list(self.finance_event_hierarchy.keys())
        prompt = f"""
        假设你是一个金融领域的事件分类模型，请分析以下新闻中公司"{company}"涉及的所有一级事件类型。
        注意：可能存在多个事件，请全部列出。
        必须从以下列表中选择：{coarse_events}
        只返回 JSON 数组格式的事件类型列表，不要回答多余的话。

        金融新闻：{news_text}
        """
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)

        # 解析事件列表
        try:
            events = _parse_items(response, coarse_events)
            if events:
                return events
        except:
            pass

        # 备选：按逗号分割并过滤
        events = _parse_items(response, coarse_events)
        return events if events else [self._fallback_extract_coarse_event(news_text, company)]

    def _fallback_extract_coarse_event(self, news_text: str, company: str) -> str:
        """备选方案：提取单个一级事件"""
        coarse_events = list(self.finance_event_hierarchy.keys())
        prompt = f"""
        从以下新闻中判断公司"{company}"涉及的一级事件类型，从{coarse_events}中选择一个。
        只回答事件类型，不要多余的话。
        金融新闻：{news_text}
        """
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)
        for event in coarse_events:
            if event in response:
                return event
        return coarse_events[0]  # 默认返回第一个

    def step3_extract_fine_events(self, news_text: str, company: str, coarse_event: str) -> List[str]:
        """Step 3: 对公司-一级事件分支提取所有二级事件"""
        fine_events = self.finance_event_hierarchy.get(coarse_event, [])
        if not fine_events:
            return ['其他事件']

        prompt = f"""
        假设你是一个金融领域的细粒度事件分类模型，请分析以下新闻中公司"{company}"发生的一级事件"{coarse_event}"下涉及的二级事件类型。
        注意：可能存在多个二级事件，请全部列出。
        必须从以下列表中选择：{fine_events}
        只返回 JSON 数组格式的事件类型列表，不要回答多余的话。

        金融新闻：{news_text}
        """
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)

        # 解析事件列表
        try:
            events = _parse_items(response, fine_events)
            if events:
                return events
        except:
            pass

        # 备选：按逗号分割并过滤
        events = _parse_items(response, fine_events)
        return events if events else [self._fallback_extract_fine_event(news_text, company, coarse_event)]

    def _fallback_extract_fine_event(self, news_text: str, company: str, coarse_event: str) -> str:
        """备选方案：提取单个二级事件"""
        fine_events = self.finance_event_hierarchy.get(coarse_event, [])
        if not fine_events:
            return '其他事件'

        prompt = f"""
        从以下新闻中判断公司"{company}"发生的一级事件"{coarse_event}"下属于哪个二级事件，从{fine_events}中选择一个。
        只回答事件类型，不要多余的话。
        金融新闻：{news_text}
        """
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)
        for event in fine_events:
            if event in response:
                return event
        return fine_events[0]  # 默认返回第一个

    def step4_extract_sentiment(self, news_text: str, company: str, coarse_event: str, fine_event: str) -> str:
        """Step 4: 判断具体事件的情感极性"""
        prompt = f"""
        假设你是一个金融领域的细粒度情感分析模型，请基于以下新闻内容，判断公司"{company}"发生的"{coarse_event}"-"{fine_event}"事件对该公司的影响是正面、中立还是负面。
        从以下情感极性列表中选择：{self.sentiments}
        只返回情感极性，不要回答多余的话。

        金融新闻：{news_text}
        """
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)
        return _normalize_sentiment(response)

    def analyze_news_tree(self, news_text: str) -> Dict[str, Any]:
        """完整的树形展开分析流程，返回结构化结果"""
        results = []
        analysis_tree = {
            'news_text': news_text,
            'companies': {},
            'total_quads': 0
        }

        # Step 1: 识别所有公司
        companies = self.step1_extract_companies(news_text)
        print(f"Step 1 - 识别到的公司: {companies}")

        # 对每家公司展开分析
        for company in companies:
            company_data = {
                'coarse_events': {}
            }

            # Step 2: 提取该公司所有一级事件
            coarse_events = self.step2_extract_coarse_events(news_text, company)
            print(f"Step 2 - 公司 {company} 的一级事件: {coarse_events}")

            # 对每个一级事件分支展开
            for coarse_event in coarse_events:
                event_data = {
                    'fine_events': {}
                }

                # Step 3: 提取该分支所有二级事件
                fine_events = self.step3_extract_fine_events(news_text, company, coarse_event)
                print(f"Step 3 - {company}-{coarse_event} 的二级事件: {fine_events}")

                # 对每个二级事件分支展开
                for fine_event in fine_events:
                    # Step 4: 判断情感极性
                    sentiment = self.step4_extract_sentiment(news_text, company, coarse_event, fine_event)
                    quad = (company, coarse_event, fine_event, sentiment)
                    results.append(quad)
                    print(f"Step 4 - 生成四元组: {quad}")

                    # 存储到树形结构
                    event_data['fine_events'][fine_event] = {'sentiment': sentiment}

                company_data['coarse_events'][coarse_event] = event_data

            analysis_tree['companies'][company] = company_data

        # 记录统计信息
        analysis_tree['total_quads'] = len(results)
        analysis_tree['quad_tuples'] = results

        # 去重（如果存在相同的四元组）
        unique_quads = _dedupe_preserve_order(results)
        if len(unique_quads) < len(results):
            print(f"去重后四元组数量: {len(unique_quads)}")
        analysis_tree['quad_tuples_unique'] = unique_quads

        return analysis_tree

    def analyze_news_with_industry(self, news_text: str) -> List[Tuple[str, str, str, str, str]]:
        """补充行业信息，生成最终五元组"""
        quads = self.analyze_news_tree(news_text)['quad_tuples_unique']
        results = []

        for company, coarse, fine, sentiment in quads:
            # 获取行业信息（此处可扩展为真实公司-行业映射）
            industry = self.company_industry_map.get(company, '未知行业')
            results.append((company, industry, coarse, fine, sentiment))

        return results

    def batch_process_news(self, news_file: str, output_file: str):
        """批量处理新闻文件"""
        news_items = self._load_news_items(news_file)
        all_results = []

        for idx, item in enumerate(news_items):
            news_text = item['news_text']
            if not news_text:
                continue

            print(f"\n处理第 {idx+1} 条新闻:")

            try:
                # 执行完整分析
                result = self.analyze_news_tree(news_text)
                all_results.append({
                    'index': item.get('index', idx),
                    'data_id': item.get('data_id'),
                    'news_text': news_text,
                    'analysis_result': result
                })
            except Exception as e:
                print(f"处理第 {idx+1} 条新闻时出错: {e}")
                continue

        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(all_results, f_out, ensure_ascii=False, indent=2)

        return all_results

    def _load_news_items(self, news_file: str) -> List[Dict[str, Any]]:
        """Load EFSA data.json or a plain text news file."""
        with open(news_file, 'r', encoding='utf-8') as f:
            content = f.read()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            items = []
            for idx, line in enumerate(content.splitlines()):
                if not line.strip():
                    continue
                items.append({
                    'index': idx,
                    'news_text': line.split('####')[0].strip()
                })
            return items

        if isinstance(data, list):
            return [
                {
                    'index': idx,
                    'data_id': record.get('data_id'),
                    'news_text': record.get('content', '').strip()
                }
                for idx, record in enumerate(data)
                if isinstance(record, dict)
            ]

        raise ValueError("news_file must be EFSA data.json or a plain text file")

def main():
    """主函数"""
    # 初始化分析器
    analyzer = FinancialEventReasoningAnalyzer("model-name")  # 替换为实际的模型名称

    # 示例1: 单条新闻分析
    news_text = """浦东建设公告,近日公司子公司上海市浦东新区建设(集团)有限公司、上海浦东路桥(集团)有限公司中标多项重大工程项目,中标金额总计15.66亿元。"""

    print("="*50)
    print("单条新闻分析示例")
    print("="*50)

    # 执行完整的树形展开分析
    result = analyzer.analyze_news_tree(news_text)

    print("\n结构化分析结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n最终四元组列表:")
    for quad in result['quad_tuples_unique']:
        print(f"('{quad[0]}', '{quad[1]}', '{quad[2]}', '{quad[3]}')")

    # 示例2: 补充行业信息，生成五元组
    print("\n" + "="*50)
    print("生成五元组（包含行业信息）")
    print("="*50)

    # 添加一些公司-行业映射示例
    analyzer.company_industry_map = {
        '浦东建设': '建筑',
        '上海市浦东新区建设(集团)有限公司': '建筑',
        '上海浦东路桥(集团)有限公司': '建筑'
    }

    five_tuples = analyzer.analyze_news_with_industry(news_text)
    for company, industry, coarse, fine, sentiment in five_tuples:
        print(f"('{company}', '{industry}', '{coarse}', '{fine}', '{sentiment}')")

    # 示例3: 批量处理（如果存在文件）
    # 注意：需要实际的文件路径
    # all_results = analyzer.batch_process_news("news_input.txt", "analysis_output.json")
    # print(f"批量处理完成，共处理 {len(all_results)} 条新闻")

if __name__ == "__main__":
    main()
