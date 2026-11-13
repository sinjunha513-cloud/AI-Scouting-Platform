import ast
import io
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from analysis_utils import standardize_metrics, recommend_players, get_player_history
from sample_data import make_sample_data


def app_functions():
    tree = ast.parse(Path('app.py').read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
             or isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) for t in n.targets)
             and isinstance(n.value, (ast.Dict, ast.List, ast.Set))]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'app.py', 'exec'), namespace)
    return namespace


class AnalysisTests(unittest.TestCase):
    def test_roles_are_independent_of_score_weights(self):
        ns = app_functions()
        data = make_sample_data()
        data['POSITION_GROUP'] = data['포지션'].apply(ns['classify_position_group'])
        for mode in ['누적 데이터', '90분당 데이터']:
            original = ns['apply_model'](data, mode, ns['base_attack_weights'], ns['base_midfielder_weights'], ns['base_defender_weights'])
            changed = ns['apply_model'](data, mode, {'AIR_CHALG_WON': 1}, ns['base_midfielder_weights'], ns['base_defender_weights'])
            pd.testing.assert_frame_equal(original[['선수명', '연도', '포지션', '복합_스타일']],
                                          changed[['선수명', '연도', '포지션', '복합_스타일']])
            attacks = original[original.POSITION_GROUP.eq('ATTACK')]
            col = 'Per_AIR_CHALG_WON_Z' if mode == '90분당 데이터' else 'AIR_CHALG_WON_Z'
            self.assertTrue(attacks[col].notna().all())
            self.assertGreater(attacks[col].abs().sum(), 0)

    def test_per90_derived_without_cumulative_fallback(self):
        ns = app_functions()
        raw = pd.DataFrame({'출전시간': [900, 1800, 0], 'Goal_Qty': [10, 20, 1],
                            'PCNT_TACKLE_WON': [60, 70, 50], 'age': [20, 25, 30]})
        prepared = ns['prepare_metric_data'](raw)
        np.testing.assert_allclose(prepared.Per_Goal_Qty.iloc[:2], [1, 1])
        self.assertTrue(pd.isna(prepared.Per_Goal_Qty.iloc[2]))
        weights = ns['convert_weights_by_mode']({'Goal_Qty': 1, 'PCNT_TACKLE_WON': 1, 'age': 1}, '90분당 데이터', prepared)
        self.assertEqual(set(weights), {'Per_Goal_Qty', 'PCNT_TACKLE_WON'})
        self.assertEqual(ns['convert_weights_by_mode']({'Goal_Qty': 1}, '90분당 데이터', raw), {})

    def test_missing_records_preserved_and_not_strengths(self):
        ns = app_functions()
        upload = io.BytesIO('year,team_name,player_name,POSITION_NM,work_time,Goal_Qty,BALL_MISS\n2025,A,a,FW,900,,\n2025,B,b,FW,900,0,0\n'.encode())
        upload.name = 'missing.csv'
        loaded = ns['load_data'].__wrapped__(upload)
        self.assertTrue(pd.isna(loaded.loc[0, 'BALL_MISS']))
        self.assertEqual(loaded.loc[1, 'BALL_MISS'], 0)
        data = pd.DataFrame({'Goal_Qty': [4, 8, np.nan], 'BALL_MISS': [1, 9, np.nan]})
        result = ns['analyze_strength_weakness'](ns['add_zscores'](data, {'Goal_Qty': 1, 'BALL_MISS': 1}),
                                               {'Goal_Qty': .5, 'BALL_MISS': .5}, 'ATTACK')
        self.assertTrue(pd.isna(result.loc[2, 'Goal_Qty']))
        self.assertEqual(result.loc[2, 'Goal_Qty_Z'], 0)
        self.assertEqual(result.loc[2, '강점'], '특이 강점 없음')
        self.assertEqual(result.loc[2, '약점'], '특이 약점 없음')

    def test_no_metrics_or_zero_weights_do_not_produce_scores(self):
        ns = app_functions()
        data = pd.DataFrame({'연도': ['2025']*2, '팀명': ['A','B'], '선수명': ['a','b'],
                             '포지션': ['FW']*2, '출전시간': [900]*2, 'POSITION_GROUP': ['ATTACK']*2})
        for mode in ['누적 데이터', '90분당 데이터']:
            self.assertTrue(ns['apply_model'](data, mode, ns['base_attack_weights'], {}, {}).empty)
        data['Goal_Qty'] = [1, 2]
        self.assertTrue(ns['apply_model'](data, '누적 데이터', {'Goal_Qty': 0}, {}, {}).empty)

    def test_strength_weakness_ranks_by_favorable_direction(self):
        ns = app_functions()
        # a: foul ratio 0 versus goal ratio 1.6 -> fewer fouls is stronger.
        # b: foul ratio 2 versus goal ratio 0.4 -> excess fouls is weaker.
        data = pd.DataFrame({'Goal_Qty': [8, 2], 'FOUL': [0, 10]})
        result = ns['analyze_strength_weakness'](data, {'Goal_Qty': .5, 'FOUL': .5}, 'ATTACK')
        self.assertEqual(result.iloc[0]['강점'].split(', ')[0], '파울')
        self.assertEqual(result.iloc[1]['약점'].split(', ')[0], '파울')

    def test_screen_handles_no_metrics(self):
        from streamlit.testing.v1 import AppTest
        source = Path('app.py').read_text(encoding='utf-8')
        minimal = "pd.DataFrame({'연도':['2025']*2,'팀명':['A','B'],'선수명':['a','b'],'포지션':['FW']*2,'출전시간':[900]*2})"
        source = source.replace('df = make_sample_data() if use_sample else load_data(uploaded_file)',
                                'df = ' + minimal + ' if use_sample else load_data(uploaded_file)')
        screen = AppTest.from_string(source, default_timeout=30).run()
        self.assertFalse(screen.exception)
        self.assertTrue(screen.warning)
        self.assertIn('평가 가능한 지표가 없습니다', screen.warning[0].value)

    def test_screen_default_modes_and_zero_weights(self):
        from streamlit.testing.v1 import AppTest
        screen = AppTest.from_file('app.py', default_timeout=45).run()
        self.assertFalse(screen.exception)
        self.assertEqual(screen.title[0].value, '⚽ AI 스카우팅 플랫폼')
        screen.radio[0].set_value('90분당 데이터').run()
        self.assertFalse(screen.exception)
        for widget in screen.slider:
            if widget.key and widget.key.startswith(('att_', 'mid_', 'def_')):
                widget.set_value(0.0)
        screen.run()
        self.assertFalse(screen.exception)
        self.assertTrue(screen.warning)

    def test_only_observed_rows_scored_and_missing_roles_not_invented(self):
        ns = app_functions()
        data = pd.DataFrame({'연도': ['2025']*3, '팀명': ['A']*3, '선수명': ['a','b','c'],
                             '포지션': ['FW']*3, '출전시간': [900]*3, 'POSITION_GROUP': ['ATTACK']*3,
                             'BALL_MISS': [1, 4, np.nan]})
        result = ns['apply_model'](data, '누적 데이터', {'BALL_MISS': 1}, {}, {})
        self.assertEqual(result['선수명'].tolist(), ['a', 'b'])
        self.assertTrue(result['수식어'].eq('정보 부족').all())
        self.assertTrue(result['포지션'].eq('FW').all())

    def test_scale_and_invalid_values(self):
        frame = pd.DataFrame({'a': [1, 2, 4], 'b': [20, 3, 8], 'constant': [1]*3})
        scaled = standardize_metrics(frame, list(frame))
        frame['a'] *= 10000
        np.testing.assert_allclose(scaled, standardize_metrics(frame, list(frame)))
        self.assertNotIn('constant', scaled)
        frame['a'] = frame['a'].astype(float)
        frame.loc[0, 'a'] = np.inf
        self.assertTrue(np.isfinite(standardize_metrics(frame, list(frame))).all().all())

    def test_recommendations_ignore_index_and_cross_season(self):
        data = pd.DataFrame({'선수명': ['A','B','C','D'], '연도': ['2025','2025','2024','2025'],
                             '팀명': ['X']*4, 'POSITION_GROUP': ['ATTACK']*4, '포지션': ['ST']*4,
                             'CLUSTER': [0,0,0,1], 'SCOUTING_SCORE': [50]*4,
                             'ML_x': [1,1,1,-1], 'ML_y': [2,2,2,-2]}, index=[7,2,9,1])
        result = recommend_players(data, data.iloc[0])
        self.assertEqual(result['선수명'].tolist(), ['B','D'])
        self.assertAlmostEqual(result.iloc[0]['코사인 유사도'], 1)
        self.assertTrue(recommend_players(data.assign(ML_x=0, ML_y=0), data.assign(ML_x=0, ML_y=0).iloc[0]).empty)

    def test_model_and_history(self):
        ns = app_functions()
        data = make_sample_data()
        data['POSITION_GROUP'] = data['포지션'].apply(ns['classify_position_group'])
        for mode in ['누적 데이터', '90분당 데이터']:
            result = ns['apply_model'](data, mode, ns['base_attack_weights'], ns['base_midfielder_weights'], ns['base_defender_weights'])
            self.assertEqual(len(result), len(data))
            self.assertEqual(get_player_history(result, data.iloc[0]['선수명'])['연도'].nunique(), 4)
            self.assertTrue(np.isfinite(result['SCOUTING_SCORE']).all())
            features = [c for c in result if c.startswith('ML_')]
            if mode == '90분당 데이터':
                self.assertTrue(all(c.startswith('ML_Per_') for c in features))
            for _, cohort in result.groupby(['연도', 'POSITION_GROUP']):
                self.assertTrue(np.allclose(cohort[features].mean().dropna(), 0, atol=1e-10))

    def test_local_workbook(self):
        files = list(Path('.').glob('*.xlsx'))
        if not files:
            self.skipTest('Private workbook not included')
        ns = app_functions()
        file = io.BytesIO(files[0].read_bytes())
        file.name = files[0].name
        data = ns['load_data'](file)
        self.assertTrue({'연도','팀명','선수명','출전시간'}.issubset(data))
        data['POSITION_GROUP'] = data['포지션'].apply(ns['classify_position_group'])
        data = data[data['출전시간'] >= 300]
        result = ns['apply_model'](data, '90분당 데이터', ns['base_attack_weights'], ns['base_midfielder_weights'], ns['base_defender_weights'])
        self.assertFalse(result.empty)
        for _, group in result.groupby(['연도', 'POSITION_GROUP']):
            target = group.iloc[0]
            rec = recommend_players(result, target)
            if not rec.empty:
                self.assertTrue((rec['선수명'] != target['선수명']).all())
                self.assertTrue((rec['연도'] == target['연도']).all())


if __name__ == '__main__':
    unittest.main()
