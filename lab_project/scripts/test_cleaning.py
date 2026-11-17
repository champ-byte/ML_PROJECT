"""Guard against damaging real names and useful address components."""

import unittest

from blocking import TfidfBlocker, candidate_map, normalize_text
from cleaning import audit, clean_for_retrieval, normalizer
from test_blocking import frame
from run_cleaning import paired_changes


class CleaningTests(unittest.TestCase):
    def test_remove_only_placeholder_components_keep_address_content(self):
        self.assertEqual(clean_for_retrieval('12 Main Rd, <NULL>, N/A, null, Delhi','business_address'), '12 main rd delhi')
        self.assertEqual(clean_for_retrieval('12 Main <NULL> Rd','business_address'),'12 main rd')
        self.assertEqual(clean_for_retrieval('<NULL>','business_address'),'')

    def test_preserve_legitimate_null_na_text(self):
        for name in ['Null Industries','THE NULL INDUSTRIES','N/A Science Inc']:
            self.assertEqual(clean_for_retrieval(name,'business_name'),normalize_text(name))
        self.assertEqual(clean_for_retrieval('Null Road, N/A Building, Nulltown','business_address'),'null road n a building nulltown')

    def test_specific_name_marker_without_blanket_angle_deletion(self):
        self.assertEqual(clean_for_retrieval('Infra Farmers <CITY_NAME>','business_name'),'infra farmers')
        self.assertEqual(clean_for_retrieval('Shop <Real Name>','business_name'),'shop real name')

    def test_field_specific_abbreviations_and_ambiguous_words(self):
        variant='placeholders_abbreviations'
        self.assertEqual(clean_for_retrieval('ABC Pvt. Ltd. Corp Inc','business_name',variant),'abc private limited corporation incorporated')
        self.assertEqual(clean_for_retrieval('1 St Rd, Apt 4, No 2','business_address',variant),'1 st road apartment 4 no 2')
        self.assertEqual(clean_for_retrieval('Rd Inc','business_name',variant),'rd incorporated')
        self.assertEqual(clean_for_retrieval('Corporation Incorporated','business_name',variant),'corporation incorporated')

    def test_preserve_unicode_joiners_accents_and_idempotence(self):
        value='CAFÉ राम\u200cमार्केट Pvt Ltd <CITY_NAME>'
        cleaned=clean_for_retrieval(value,'business_name','placeholders_abbreviations')
        self.assertIn('\u200c',cleaned)
        self.assertIn('café',cleaned)
        self.assertEqual(clean_for_retrieval(cleaned,'business_name','placeholders_abbreviations'),cleaned)

    def test_restrict_fields_and_variants(self):
        with self.assertRaises(ValueError):clean_for_retrieval('US','country')
        with self.assertRaises(ValueError):clean_for_retrieval('ABC','business_name','unfrozen_variant')

    def test_audit_detects_changed_fields_and_protected_names(self):
        sources={'source1':frame([('S1-a','Null Industries','1 Rd, <NULL>','US')])}
        tables=audit(sources)
        self.assertEqual(tables['address_components']['occurrences'].sum(),1)
        self.assertEqual(tables['protected_names']['raw_name'].to_list(),['Null Industries'])
        affected=tables['counts'].filter(tables['counts']['issue']=='placeholder_cleanup_changes_view')
        self.assertEqual(affected['affected_records'].sum(),1)

    def test_cleaner_is_applied_consistently_in_fit_and_prepare(self):
        sources={'source1':frame([('S1-a','Alpha Pvt Ltd','1 Rd, NULL','India')]),
                 'source2':frame([('S2-a','Alpha Private Limited','1 Road','India')]),
                 'source3':frame([])}
        blocker=TfidfBlocker(normalizer=normalizer('placeholders_abbreviations')).fit(sources)
        prepared=blocker.prepare(sources)
        self.assertEqual(prepared.normalized['source1']['name'],prepared.normalized['source2']['name'])
        self.assertEqual(prepared.normalized['source1']['address'],prepared.normalized['source2']['address'])
        evidence,_=blocker.retrieve(prepared,10)
        self.assertEqual(candidate_map(['S1-a'],evidence),{'S1-a':{'S2-a'}})

    def test_paired_diagnostics_expose_offsetting_gain_and_loss(self):
        sources={'source1':frame([('S1-a','Alpha','1 Road','US')]),
                 'source2':frame([('S2-a','Alpha','1 Road','US'),('S2-b','Alpha','1 Road','US')]),
                 'source3':frame([])}
        per_query,churn,changed=paired_changes({'S1-a':{'S2-a','S2-b'}},
            {'S1-a':{'S2-a'}},{'S1-a':{'S2-b'}},sources,'placeholders')
        self.assertEqual(per_query['recovered_true_pairs'].sum(),1)
        self.assertEqual(per_query['newly_lost_true_pairs'].sum(),1)
        self.assertEqual(set(changed['change']),{'recovered','newly_lost'})
        self.assertTrue(all(churn['is_true_match']))


if __name__=='__main__':unittest.main()
