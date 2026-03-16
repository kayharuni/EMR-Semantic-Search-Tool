import pandas as pd
import numpy as np
import functools as ft
import re
from sentence_transformers import SentenceTransformer, util
import itertools as it
import math


class SemanticEngine:
    def __init__(self, data_path: str, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.data_path = data_path
        self.model = SentenceTransformer(model_name)

        # load + prepare everything once
        self._load_data()
        self._prepare_emr()
        self._prepare_master_embedding()
        self._tumor_icd_pattern()

    # ==========================================
    # DATA LOADING 
    # ==========================================

    def _load_data(self):
        TO_REMOVE = ['CREATED_BY', 'CREATED_DT', 'UPDATED_BY', 'UPDATED_DT']

        self.checkup = pd.read_csv(f"{self.data_path}/tb_m_checkup.csv").drop(
            ['CREATED_BY', 'CREATED_DT'], axis=1, errors="ignore"
        )

        self.mcu = pd.read_csv(f"{self.data_path}/tb_r_mcu.csv", sep="\t").drop(
            TO_REMOVE, axis=1, errors="ignore"
        )

        self.lab = pd.read_csv(f"{self.data_path}/tb_r_lab.csv", sep="\t").drop(
            TO_REMOVE, axis=1, errors="ignore"
        )

        self.physics = pd.read_csv(f"{self.data_path}/tb_r_physics.csv", sep="\t").drop(
            ['CREATED_BY', 'CREATED_DT'], axis=1, errors="ignore"
        )

        self.radiology = pd.read_csv(f"{self.data_path}/tb_r_radiology.csv", sep="\t").drop(
            TO_REMOVE, axis=1, errors="ignore"
        )

        self.anamnesa = pd.read_csv(f"{self.data_path}/tb_r_anamnesa.csv", sep="\t").drop(
            TO_REMOVE, axis=1, errors="ignore"
        )

        self.cardio = pd.read_csv(f"{self.data_path}/tb_r_cardio.csv", sep="\t").drop(
            TO_REMOVE, axis=1, errors="ignore"
        )

        self.papsmear = pd.read_csv(f"{self.data_path}/tb_r_papsmear.csv", sep="\t").drop(
            TO_REMOVE, axis=1, errors="ignore"
        )
        
        self.master_df = pd.read_csv(f"{self.data_path}/master_data2.csv", index_col=0).reset_index(drop=True)

        self.lab_check = pd.merge(
            self.lab, self.checkup, on='CHECKUP_ID', how='left'
        )

        self.hospi = pd.read_csv(f"{self.data_path}/tb_r_hospitalization.csv").drop(
            ['CREATED_BY', 'CREATED_DT', 'UPDATED_BY', 'UPDATED_DT', 'PATIENT_NAME'], axis=1, errors="ignore"
        )

    # ==========================================
    # PREPARE TUMOR ICD
    # ==========================================

    def _tumor_icd_pattern(self):
        tumor_icd = []
        c_ranges = it.chain(range(0, 27), range(30, 41), 
                            range(43, 59), range(60, 76),
                            range(76, 96))

        tumor_icd += [rf'^C{i:02d}' for i in c_ranges]
        tumor_icd += [rf'^D{i:02}' for i in it.chain(range(50))]
        tumor_icd.append(r'C7A')
        tumor_icd.append(r'C7B')
        self.tumor_icd_pattern = "|".join(tumor_icd)

    # ==========================================
    # PREPARE EMR
    # ==========================================

    @staticmethod
    def pseudo_id(series, prefix="P"):
        unique_vals = series.unique()
        mapping = {val: f"{prefix}{i:03d}" for i, val in enumerate(unique_vals)}
        return series.replace(mapping)

    def _build_emr(self):
        to_merge = [
            self.mcu,
            self.physics,
            self.cardio,
            self.radiology,
            self.anamnesa,
            self.papsmear
        ]

        df_emr = ft.reduce(
            lambda left, right: pd.merge(left, right, how="outer", on="TRANS_NO"),
            to_merge
        )

        return df_emr
    
    def _prepare_emr(self):
        self.mcu["EMPLOYEE_ID"] = self.pseudo_id(self.mcu["EMPLOYEE_ID"])

        self.df_emr = self._build_emr()

        self.df_emr = self.df_emr[['EMPLOYEE_ID', 'TRANS_NO', 'PROVIDER_CODE','TRANS_DT', 'NUTRITION_STS','SKIN_DISEASES','SKIN_DISEASES_OTHER',
                       'R_EYE_VISUS_STS_AFTER', 'L_EYE_VISUS_STS_AFTER',
                       'R_EAR_FUNCTION_STS','L_EAR_FUNCTION_STS',  'RHYTHM_STS', 'TONSIL_DESC', 'HERNIA_DESC', 'TIROID_DESC', 
                       'GENITAL_DESC', 'NECK_DESC', 'AXILA_DESC','INGUINAL_DESC', 'DESC_ECHO', 'USG_CONCLUSION_PAYUDARA',
                       'USG_FOUND_GINEKOLOGI','USG_RECOMMENDATION_GINEKOLOGI', 'MAMO_FOUND','MAMO_CONCLUSION', 'USG_CONCLUSION_GINEKOLOGI',
                        'HEART_LINE_STS', 'L_EAR_MEMBRAN_STATUS_DESC', 'R_EAR_MEMBRAN_STATUS_DESC',
                        'RECTAL_DESC', 'DESC_EKG', 'DESC_TREADMILL', 'SUGGEST_EKG', 'SUGGEST_TREADMILL', 'RO_IMPRESSION',
                        'RO_INFORMATION','RO_INFO_CHECKBOX','RO_INFO_OTHER','USG_FOUND_ABDOMEN','USG_CONCLUSION_ABDOMEN',
                        'USG_RECOMMENDATION_ABDOMEN','HIST_DISEASE','HIST_HOSPITALIZATION','HIST_SURGERY',
                        'HIST_ACCIDENT', 'CURR_DISEASE','CURR_MEDICINE','CURR_MEDICATION', 'CURR_MEDICATION_DURATION',
                        'CURR_MEDICATION_MEDICINE','FAM_HYPERTENSI','FAM_DIABETES','FAM_DIABETES_OTHER','FAM_HEART',
                        'STATUS','DESCRIPTION','SARAN']]

        missing_df = (
            self.df_emr.isnull().sum() / len(self.df_emr) * 100
        ).reset_index()

        columns_drop = missing_df[
            missing_df[0] == 100
        ]["index"].to_list()

        self.df_emr_clean = self.df_emr.drop(columns_drop, axis=1)

        # file trace map
        self.files = {
            "mcu": set(self.mcu.columns),
            "physics": set(self.physics.columns),
            "radiology": set(self.radiology.columns),
            "anamnesa": set(self.anamnesa.columns),
            "cardio": set(self.cardio.columns),
            "papsmear": set(self.papsmear.columns),
            "hospitalization": set(self.hospi.columns)
        }

    # ==========================================
    # MASTER EMBEDDING
    # ==========================================

    def _prepare_master_embedding(self):
        #self.master_df = self.master_df.fillna("")

        self.master_list = [words.lower() for words in self.master_df['to_embed'].tolist()]

        self.master_embedding = self.model.encode(
            self.master_list,
            convert_to_tensor=True
        )

    # ==========================================
    # POSITIVE SENTENCE FILTER
    # ==========================================

    @staticmethod
    def get_positive_sentences(text_list, query):
        query = query.lower()
        safe_query = re.escape(query)

        neg_patterns = [
            r"(?<!not\s)(?<!tidak\s)\bnormal\b",
            r"\btidak tampak kelainan\b",
            r"\bnegative\b",
            r"\btidak ada\b",
            r"\btidak ada masalah\b",
            r"\btidak ada riwayat\b"
        ]

        if safe_query:
            neg_patterns.append(rf"\bno {safe_query}\b")
            neg_patterns.append(rf"\btidak ada {safe_query}\b")

        combined = "|".join(neg_patterns)
        results = []

        for text in text_list:
            sentences = re.split(r'[.;]', str(text))
            valid = [s.strip() for s in sentences if not re.search(combined, s, re.I)]
            results.append(". ".join(valid))

        return results
    

    # ==========================================
    # MAIN SEARCH (FOR .NET CALL)
    # ==========================================

    def search(self, employee_id: str, query: str, top_k: int = 5, threshold: float = 0.5):
        
        if not employee_id in self.df_emr_clean['EMPLOYEE_ID'].unique().tolist():
            return {"status": "patient_not_found"}

        selected_patient = self.df_emr_clean[
            self.df_emr_clean['EMPLOYEE_ID'] == employee_id
        ]

        if selected_patient.empty:
            return {"status": "not_found"}

        all_long = []

        for _, row in selected_patient.iterrows():
            date_val = row['TRANS_DT']
            melted = row.to_frame().T.melt()
            melted["date"] = date_val
            all_long.append(melted)

        long_emr = pd.concat(all_long, ignore_index=True)

        filtered = long_emr[
            ~long_emr['variable'].isin(
                ['EMPLOYEE_ID', 'TRANS_NO', 'PROVIDER_CODE', 'TRANS_DT']
            )
        ].dropna()

        filtered["value"] = self.get_positive_sentences(
            filtered["value"],
            query
        )

        filtered = filtered[filtered["value"] != ""]

        medical_notes = filtered.value.tolist()
        columns = filtered.variable.tolist()
        dates = filtered.date.tolist()

        if not medical_notes:
            return {"status": "no_medical_notes"}

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        patient_embedding = self.model.encode(medical_notes, convert_to_tensor=True)

        hits = util.semantic_search(
            query_embedding,
            self.master_embedding,
            top_k=top_k
        )[0]

        if not hits:
            return {"status": "no_concept_match",
                    "hits": hits}
        else:
            concept_idx = pd.DataFrame(hits)['corpus_id'].iloc[0]
            concept_embedding = self.master_embedding[concept_idx]

        #if pd.DataFrame(hits)['score'].iloc[0] > threshold:
        hits2 = util.semantic_search(
            concept_embedding, patient_embedding, top_k=top_k
        )[0]
        concept = str(self.master_df.iloc[concept_idx]['concept'])
        # else: 
        #     hits2 = util.semantic_search(
        #         query_embedding, patient_embedding, top_k=top_k
        #     )[0]
        #     concept = query

        results = []

        for hit in hits2:
            if hit["score"] < threshold:
                continue

            idx = hit["corpus_id"]

            # trace file source
            source = "unknown"
            for name, cols in self.files.items():
                if columns[idx] in cols:
                    source = name
                    break

            results.append({
                "source": source,
                "feature": columns[idx],
                "transaction_date": str(dates[idx]),
                "note": medical_notes[idx],
                "similarity_score": float(hit["score"])
            })


        if not results:
            return {"status": "no_match"}

        concept_pick_row = self.master_df.iloc[concept_idx].to_dict()
        concept_icd = concept_pick_row['ICD']
        concept_checkup_cd = concept_pick_row['CHECKUP_CD']

        return {
            "status": "success" if results else "no_match",
            "concept": concept,
            "results": results,
            "concept_checkup_cd": concept_checkup_cd,
            "concept_icd": concept_icd,
              }
    
    # ==========================================
    # LAB CHECK SEARCH
    # ==========================================
     
    def labcheck_search(self, employee_id: str, concept_checkup_cd):
        
        if not employee_id in self.df_emr_clean['EMPLOYEE_ID'].unique().tolist():
            return {"status": "patient_not_found"}

        selected_patient = self.df_emr_clean[self.df_emr_clean['EMPLOYEE_ID'] == employee_id]
        if selected_patient.empty:
            return {"status": "no_patient_found"}

        trans_no_list = selected_patient['TRANS_NO'].unique().tolist()
        selected_lab = self.lab_check[self.lab_check['TRANS_NO'].isin(trans_no_list)]
        filtered_lab = selected_lab[selected_lab['CONDITION'] == 'ABNORMAL']

        if concept_checkup_cd is None:
            return {"status": "invalid_concept_data"}

        concept_checkup = [str(id).strip() for id in concept_checkup_cd.split(',')]
        labcheckup_result = filtered_lab[filtered_lab['CHECKUP_CD'].isin(concept_checkup)]

        return {
            "status": "success",
            "results": labcheckup_result.astype(str).to_dict(orient='records')
    }

    # ==============================
    # Check for ICD-10 of hospitalization
    # ==============================

    def hospi_search(self, noreg: str, concept, concept_icd):

        if not noreg in self.hospi['NOREG'].unique().tolist():
            return {"status": "No_registration_number_found"}
        
        hospi_patient = self.hospi[self.hospi['NOREG'] == noreg]
        if hospi_patient.empty:
            return("No_registration_number_found")
            
        if concept_icd is None:
            return("Invalid concept data")
        if concept == 'tumor':
            matched_icd = []
            for j in hospi_patient['ICD']:
                if re.match(self.tumor_icd_pattern, j, re.I):
                    matched_icd.append(j)
                    hospi_result = hospi_patient[hospi_patient['ICD'].isin(matched_icd)][['NOREG', 'ADMISSION_DATE','PRIMARY_DESC', 'ICD']]
                else:
                    pass
        else:
            concept_icd = [icd.strip() for icd in concept_icd.split(sep=',')]
            hospi_result = hospi_patient[hospi_patient['ICD'].isin(concept_icd)][['NOREG', 'ADMISSION_DATE','PRIMARY_DESC', 'ICD']]
        
        return {"status": "success",
                "results": hospi_result.astype(str).to_dict(orient='records')}

