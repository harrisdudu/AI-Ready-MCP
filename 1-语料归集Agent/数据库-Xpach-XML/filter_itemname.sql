-- 筛选 XML 节点 ItemName 值为 '初步诊断' 的 SQL 查询示例

-- 假设表结构如下:
-- CREATE TABLE CDR_EMR.dbo.EMR_DocList (
--     VisitNumber VARCHAR(50),
--     DocTypeName NVARCHAR(100),
--     DocContent NVARCHAR(MAX),
--     DocXML NVARCHAR(MAX)
-- );

-- 示例1: 使用 CONVERT 将 nvarchar(max) 转换为 XML 类型后筛选
SELECT 
    VisitNumber,
    DocTypeName,
    DocContent,
    DocXML,
    CONVERT(XML, DocXML).value('(/Root/ItemName)[1]', 'NVARCHAR(100)') AS ItemNameValue
FROM 
    CDR_EMR.dbo.EMR_DocList
WHERE 
    a.DocTypeName = N'入院记录'
    AND CONVERT(XML, DocXML).exist('/Root/ItemName[text() = "初步诊断"]') = 1;

-- 示例2: 使用 CTE (公共表表达式) 先转换类型再筛选
WITH XMLConverted AS (
    SELECT 
        VisitNumber,
        DocTypeName,
        DocContent,
        DocXML,
        CONVERT(XML, DocXML) AS XmlData
    FROM 
        CDR_EMR.dbo.EMR_DocList
    WHERE 
        a.DocTypeName = N'入院记录'
)
SELECT 
    VisitNumber,
    DocTypeName,
    DocContent,
    DocXML,
    XmlData.value('(/Root/ItemName)[1]', 'NVARCHAR(100)') AS ItemNameValue
FROM 
    XMLConverted
WHERE 
    XmlData.exist('/Root/ItemName[text() = "初步诊断"]') = 1;

-- 示例3: 如果 DocXML 列已经是 XML 类型，可以直接使用
/*
SELECT 
    VisitNumber,
    DocTypeName,
    DocContent,
    DocXML,
    DocXML.value('(/Root/ItemName)[1]', 'NVARCHAR(100)') AS ItemNameValue
FROM 
    CDR_EMR.dbo.EMR_DocList
WHERE 
    a.DocTypeName = N'入院记录'
    AND DocXML.exist('/Root/ItemName[text() = "初步诊断"]') = 1;
*/

-- 注意事项:
-- 1. 确保 DocXML 中的内容是有效的 XML 格式
-- 2. XPath 表达式区分大小写，请确保节点名称与实际 XML 中的一致
-- 3. 如果 XML 结构不同，请调整 XPath 表达式以匹配实际结构
-- 4. 对于大型数据集，考虑在转换后的 XML 列上创建索引以提高性能