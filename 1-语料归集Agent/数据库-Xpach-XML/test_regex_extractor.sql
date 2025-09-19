-- 测试正则表达式数据提取器
-- 适用于 SQL Server 数据库

-- 确保正则表达式提取器已创建
-- 运行此脚本前，请先运行 regex_data_extractor.sql

-- 清除现有模式（如果存在）
DELETE FROM RegexPatterns;
GO

-- 添加测试模式
EXEC AddRegexPattern
    @PatternName = 'patient_name',
    @Pattern = '%患者姓名[:：]%[吖-座]%',
    @Description = '提取患者姓名';

EXEC AddRegexPattern
    @PatternName = 'gender',
    @Pattern = '%性别[:：]%[男女]%',
    @Description = '提取性别';

EXEC AddRegexPattern
    @PatternName = 'age',
    @Pattern = '%年龄[:：]%[0-9]%',
    @Description = '提取年龄';

EXEC AddRegexPattern
    @PatternName = 'visit_number',
    @Pattern = '%就诊号[:：]%[A-Za-z0-9]%',
    @Description = '提取就诊号';

EXEC AddRegexPattern
    @PatternName = 'diagnosis',
    @Pattern = '%诊断[:：]%[吖-座、,；;]%',
    @Description = '提取诊断信息';

-- 创建测试结果表
CREATE TABLE #TestResults (
    PatternName NVARCHAR(100),
    ExtractedValue NVARCHAR(MAX)
);

-- 测试用例1：基本患者信息
DECLARE @SampleText1 NVARCHAR(MAX) = N'
患者信息：
患者姓名：张三
性别：男
年龄：45岁
就诊号：B000831020
诊断：高血压、糖尿病
';

-- 提取数据
INSERT INTO #TestResults
SELECT * FROM ExtractFromText(@SampleText1);

-- 查询结果
SELECT '测试用例1' AS TestCase, * FROM #TestResults;

-- 清空测试表
TRUNCATE TABLE #TestResults;

-- 测试用例2：不同格式的患者信息
DECLARE @SampleText2 NVARCHAR(MAX) = N'
姓名：李四
性别: 女
年龄：65
就诊编号：A123456789
诊断结果：冠心病；高脂血症
';

-- 提取数据
INSERT INTO #TestResults
SELECT * FROM ExtractFromText(@SampleText2);

-- 查询结果
SELECT '测试用例2' AS TestCase, * FROM #TestResults;

-- 测试用例3：含有日期的医疗记录
DECLARE @SampleText3 NVARCHAR(MAX) = N'
患者：王五
性别：男
年龄：50岁
就诊日期：2024-05-20
诊断：肺炎
治疗方案：抗生素治疗
';

-- 添加日期模式
EXEC AddRegexPattern
    @PatternName = 'date',
    @Pattern = '%[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]%',
    @Description = '提取日期';

-- 提取数据
TRUNCATE TABLE #TestResults;
INSERT INTO #TestResults
SELECT * FROM ExtractFromText(@SampleText3);

-- 查询结果
SELECT '测试用例3' AS TestCase, * FROM #TestResults;

-- 清理
DROP TABLE #TestResults;

-- 说明：
-- 1. 此测试脚本演示了如何使用正则表达式数据提取器
-- 2. 测试了不同格式的医疗文本数据提取
-- 3. 结果可能因SQL Server的PATINDEX函数的局限性而有所不同
-- 4. 对于更复杂的提取需求，可能需要调整模式或考虑使用CLR集成