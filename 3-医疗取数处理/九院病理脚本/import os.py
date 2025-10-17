import os
import pandas as pd
import pyodbc
from tqdm import tqdm
from datetime import datetime

# === 日志系统 ===
LOG_PATH = 'process_log.txt'
def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")


visitnum_txt_path = 'visit_numbers.txt'
visit_numbers_s = set()
with open(visitnum_txt_path) as f:
    for l in f.readlines():
        visit_numbers_s.add(l.strip())    
visit_numbers = list(visit_numbers_s)


# === 数据库连接函数 ===
SERVER = '172.28.10.1'
USERNAME = 'sa'
PASSWORD = 'Kps@123456!'
DRIVER = '{ODBC Driver 17 for SQL Server}'

# 数据库连接配置


def get_conn(database):
    try:
        conn_str = (
            f'DRIVER={DRIVER};'
            f'SERVER={SERVER};'
            f'DATABASE={database};'
            f'UID={USERNAME};'
            f'PWD={PASSWORD}'
        )
        conn = pyodbc.connect(conn_str)
        return conn
    except Exception as e:
        log(f"❌ 连接数据库 {database} 失败: {e}")
        return None

# === 获取所有的 empid === 
def get_all_empid():
    placeholders = ','.join(['?'] * len(visit_numbers))
    conn_mdm = get_conn('CDR_MDM')
    if conn_mdm:
        query_empi = f"""
            SELECT CDRVisitNumber, EmpiDisplayID, EMPID FROM MDM_EMPI_VisitNumber 
            WHERE CDRVisitNumber IN ({placeholders})
        """
        df_cdr_mdm = pd.read_sql(query_empi, conn_mdm, params=visit_numbers)
        conn_mdm.close()



# === 主处理流程 ===
for visit_number in tqdm(visit_numbers, desc="🏥 正在处理住院号"):
    log(f"\n🔎 开始处理住院号: {visit_number}")
    try:
        output_dir = f'out/{visit_number}'
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        log(f"❌ 创建目录失败: {e}")
        continue

    # === 1. 查询 EMR_DocList ===
    try:
        conn_emr = get_conn('CDR_EMR')
        if conn_emr:
            query_emr = """
                SELECT * FROM EMR_DocList 
                WHERE isDeleted = 0
                  AND VisitNumber = ?
            """
            df_emr = pd.read_sql(query_emr, conn_emr, params=[visit_number])
            conn_emr.close()
            if not df_emr.empty:
                log(f"✅ EMR_DocList 数据已获取（{len(df_emr)} 行）")
                
                # 单独保存DocXML字段内容为文本文件
                if 'DocXML' in df_emr.columns:
                    docxml_dir = os.path.join(output_dir, 'DocXML_Files')
                    os.makedirs(docxml_dir, exist_ok=True)
                    
                    docxml_count = 0
                    for idx, row in df_emr.iterrows():
                        docxml_content = row.get('DocXML', '')
                        if pd.notna(docxml_content) and str(docxml_content).strip():
                            # 生成文件名：只使用DocID
                            doc_id = row.get('DocID', f'row_{idx}')
                            
                            filename = f"{doc_id}.xml"
                            filepath = os.path.join(docxml_dir, filename)
                            
                            try:
                                with open(filepath, 'w', encoding='utf-8') as f:
                                    f.write(str(docxml_content))
                                docxml_count += 1
                            except Exception as e:
                                log(f"⚠️ 保存DocXML文件失败 {filename}: {e}")
                    
                    if docxml_count > 0:
                        log(f"✅ DocXML内容已单独保存为 {docxml_count} 个文件到 {docxml_dir}")
                    else:
                        log("ℹ️ 没有找到有效的DocXML内容")
            else:
                log("⚠️ EMR_DocList 无记录")
    except Exception as e:
        log(f"❌ 查询 EMR_DocList 失败: {e}")

    # === 1.5 ， 查询 CDR_MR.dbo.MR_MedicalRecord 表，获取 VisitNumber 对应的 PatientName 
    try:
        conn_mr = get_conn('CDR_MR')
        if conn_mr:
            df_mr = pd.read_sql(
                "SELECT VisitNumber, PatientName FROM MR_MedicalRecord WHERE VisitNumber = ?",
                conn_mr, params=[visit_number])
            conn_mr.close()
            if not df_mr.empty:
                log(f"✅ MR_MedicalRecord 数据已获取（{len(df_mr)} 行）")
            else:
                log("⚠️ MR_MedicalRecord 无记录")
    except Exception as e:
        log(f"❌ 查询 MR_MedicalRecord 失败: {e}")

    # === 2. AdmissionDate & CDRVisitNumber ===
    try:
        admission_date = None

        conn_mr = get_conn('CDR_MR')
        if conn_mr:
            df_mr = pd.read_sql(
                "SELECT AdmissionDate FROM MR_MedicalRecord WHERE VisitNumber = ?",
                conn_mr, params=[visit_number])
            conn_mr.close()

            if not df_mr.empty and pd.notna(df_mr.iloc[0]['AdmissionDate']):
                admission_date = pd.to_datetime(df_mr.iloc[0]['AdmissionDate'])
                date_start = admission_date - pd.Timedelta(days=7)
                log(f"✅ AdmissionDate = {admission_date.date()}")
            else:
                log("⚠️ 无 AdmissionDate，跳过该住院号")
                continue

        # === CDRVisitNumber ===
        cdr_visits = []
        conn_mdm = get_conn('CDR_MDM')
        if conn_mdm:
            query_empi = """
                SELECT DISTINCT CDRVisitNumber FROM MDM_EMPI_VisitNumber 
                WHERE EmpiDisplayID IN (
                    SELECT EmpiDisplayID FROM MDM_EMPI_VisitNumber 
                    WHERE CDRVisitNumber = ?
                )
            """
            df_cdr_visits = pd.read_sql(query_empi, conn_mdm, params=[visit_number])
            conn_mdm.close()
            cdr_visits = df_cdr_visits['CDRVisitNumber'].dropna().unique().tolist()
            log(f"✅ 获取到 {len(cdr_visits)} 个关联 CDRVisitNumber")
        
        if not cdr_visits:
            log("⚠️ 无对应 CDRVisitNumber，跳过该住院号")
            continue

        # === 3. 查询 OBS_Report ===
        try:
            conn_obs = get_conn('CDR_OBS')
            if conn_obs:
                placeholders = ','.join(['?'] * len(cdr_visits))
                query_obs = f"""
                    SELECT obs.*
                    FROM OBS_Report obs 
                    WHERE obs.isDeleted = 0 
                      AND (
                          ( obs.ReportStatus IN (50, 60, 70)
                            AND obs.VisitNumber IN ({placeholders}) 
                            AND obs.FinalResultDateTime BETWEEN ? AND ?)
                          OR (
                            obs.ReportStatus IN (50, 60, 70)
                            AND obs.VisitNumber = ?
                          )
                      )
                """
                df_obs = pd.read_sql(query_obs, conn_obs,
                                     params=cdr_visits + [date_start, admission_date, visit_number])
                conn_obs.close()
                if not df_obs.empty:
                    log(f"✅ OBS_Report 数据已获取（{len(df_obs)} 行）")
                else:
                    log("⚠️ OBS_Report 无记录")
        except Exception as e:
            log(f"❌ 查询 OBS_Report 失败: {e}")

        # === 4. 查询 Lab_OBX 联表 Lab_OBR ===
        try:
            conn_lab = get_conn('CDR_LAB')
            if conn_lab:
                placeholders = ','.join(['?'] * len(cdr_visits))

                query_obx = f"""
                    SELECT report.req_reason, obx.*
                    FROM Lab_OBX obx
                    INNER JOIN Lab_OBR obr 
                        ON obx.FillerOrderNo = obr.FillerOrderNo
                    INNER JOIN rmlis6.dbo.lab_report  report
                        ON (obr.FeedKey='reportid' AND obr.FeedValue=cast(report.reportid as nvarchar(60)))
                    WHERE (
                            obx.isDeleted = 0
                            AND obr.isDeleted = 0
                            AND obr.VisitNumber IN ({placeholders})
                            AND obr.FinalResultDateTime BETWEEN ? AND ?
                            AND report.req_reason is not null
                          ) 
                       OR (
                            obx.isDeleted = 0
                            AND obr.isDeleted = 0
                            AND obr.VisitNumber = ?
                            AND report.req_reason is not null
                        )
                """
                df_obx = pd.read_sql(query_obx, conn_lab,
                                     params=cdr_visits + [date_start, admission_date, visit_number])

                conn_lab.close()
                if not df_obx.empty:
                    log(f"✅ Lab_OBX 数据已获取（{len(df_obx)} 行）")
                else:
                    log("⚠️ Lab_OBX 无记录")
        except Exception as e:
            log(f"❌ 查询 Lab_OBX 失败: {e}")

        # === 5. 查询病理报告 ===
        try:
            conn_mr = get_conn('CDR_MR')
            if conn_mr:
                query_obx = f"""
                    select * from PATH_Report 
                    where ReportStatus = 60
                          AND VisitNumber = ?
                          AND ClassDescription = ?
                          AND PathCategoryDescription = ?
                """
                df_mr = pd.read_sql(query_obx, conn_mr,
                                     params= [visit_number, '大体病理', '常规'])

                conn_mr.close()
                if not df_mr.empty:
                    log(f"✅ Lab_MR 数据已获取（{len(df_obx)} 行）")
                else:
                    log("⚠️ Lab_MR 无记录")
        except Exception as e:
            log(f"❌ 查询 Lab_MR 失败: {e}")

    except Exception as e:
        log(f"❌ 主处理流程异常: {e}")

log("\n🎉 所有住院号处理完成！")
