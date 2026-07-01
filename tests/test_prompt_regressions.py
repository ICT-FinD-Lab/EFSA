import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = object
    transformers.AutoModel = object
    sys.modules["transformers"] = transformers

    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


direct_prompt = load_module("direct_prompt", "code/direct_prompt.py")
reasoning_prompt = load_module("reasoning_prompt", "code/reasoning_prompt.py")


class PromptRegressionTests(unittest.TestCase):
    def test_parse_items_accepts_chinese_delimiters(self):
        values = ["经营", "股票", "合规信用"]

        self.assertEqual(
            reasoning_prompt._parse_items("经营、股票，合规信用", values),
            ["经营", "股票", "合规信用"],
        )
        self.assertEqual(
            direct_prompt._parse_items('["经营", "股票"]', values),
            ["经营", "股票"],
        )

    def test_unique_quad_key_is_always_present(self):
        analyzer = object.__new__(reasoning_prompt.FinancialEventReasoningAnalyzer)
        analyzer.step1_extract_companies = lambda text: ["浦东建设"]
        analyzer.step2_extract_coarse_events = lambda text, company: ["经营"]
        analyzer.step3_extract_fine_events = lambda text, company, coarse: ["项目中标"]
        analyzer.step4_extract_sentiment = lambda text, company, coarse, fine: "正面"

        result = analyzer.analyze_news_tree("news")

        self.assertEqual(
            result["quad_tuples_unique"],
            [("浦东建设", "经营", "项目中标", "正面")],
        )

    def test_data_json_loader_uses_content_field(self):
        analyzer = object.__new__(reasoning_prompt.FinancialEventReasoningAnalyzer)
        records = [
            {
                "data_id": "row-1",
                "content": "普路通收到政府补助。",
                "company": [
                    {
                        "company_name": "普路通",
                        "label_1": "经营",
                        "label_2": "政府补贴",
                        "sentiment": "正面",
                    }
                ],
            }
        ]

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as f:
            json.dump(records, f, ensure_ascii=False)
            f.flush()

            items = analyzer._load_news_items(f.name)

        self.assertEqual(
            items,
            [{"index": 0, "data_id": "row-1", "news_text": "普路通收到政府补助。"}],
        )

    def test_prompt_label_sets_cover_dataset_labels(self):
        with open(ROOT / "data/data.json", encoding="utf-8") as f:
            records = json.load(f)

        label_pairs = {
            (company["label_1"], company["label_2"])
            for record in records
            for company in record.get("company", [])
        }
        sentiments = {
            company["sentiment"]
            for record in records
            for company in record.get("company", [])
        }

        for module in (direct_prompt, reasoning_prompt):
            hierarchy = module.FINANCE_EVENT_HIERARCHY
            missing_labels = [
                (label_1, label_2)
                for label_1, label_2 in sorted(label_pairs)
                if label_1 not in hierarchy or label_2 not in hierarchy[label_1]
            ]

            self.assertEqual(missing_labels, [])
            self.assertTrue(sentiments.issubset(set(module.SENTIMENTS)))


if __name__ == "__main__":
    unittest.main()
