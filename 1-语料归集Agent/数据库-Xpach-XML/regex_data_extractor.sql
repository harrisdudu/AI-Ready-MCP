-- 正则表达式数据提取器 SQL 实现
-- 适用于 SQL Server 数据库

-- 创建模式存储表
CREATE TABLE RegexPatterns (
    PatternID INT IDENTITY(1,1) PRIMARY KEY,
    PatternName NVARCHAR(100) NOT NULL UNIQUE,
    Pattern NVARCHAR(500) NOT NULL,
    Description NVARCHAR(500)
);
GO

-- 添加模式存储过程
CREATE PROCEDURE AddRegexPattern
    @PatternName NVARCHAR(100),
    @Pattern NVARCHAR(500),
    @Description NVARCHAR(500) = NULL
AS
BEGIN
    -- 检查模式是否已存在
    IF EXISTS (SELECT 1 FROM RegexPatterns WHERE PatternName = @PatternName)
    BEGIN
        -- 更新现有模式
        UPDATE RegexPatterns
        SET Pattern = @Pattern,
            Description = @Description
        WHERE PatternName = @PatternName;
    END
    ELSE
    BEGIN
        -- 添加新模式
        INSERT INTO RegexPatterns (PatternName, Pattern, Description)
        VALUES (@PatternName, @Pattern, @Description);
    END
END
GO

-- 从文本提取数据函数
CREATE FUNCTION ExtractFromText
(
    @Text NVARCHAR(MAX)
)
RETURNS @Results TABLE
(
    PatternName NVARCHAR(100),
    ExtractedValue NVARCHAR(MAX)
)
AS
BEGIN
    DECLARE @PatternID INT,
            @PatternName NVARCHAR(100),
            @Pattern NVARCHAR(500),
            @StartPos INT,
            @EndPos INT,
            @MatchLength INT,
            @ExtractedValue NVARCHAR(MAX);

    -- 游标遍历所有模式
    DECLARE pattern_cursor CURSOR FOR
    SELECT PatternID, PatternName, Pattern
    FROM RegexPatterns;

    OPEN pattern_cursor;
    FETCH NEXT FROM pattern_cursor INTO @PatternID, @PatternName, @Pattern;

    WHILE @@FETCH_STATUS = 0
    BEGIN
        -- 使用 PATINDEX 查找匹配
        SET @StartPos = PATINDEX(@Pattern, @Text);
        IF @StartPos > 0
        BEGIN
            -- 提取匹配值（简化版，实际使用时可能需要根据具体模式调整）
            -- 查找模式中的第一个和最后一个通配符之间的内容
            DECLARE @WildcardStart INT = CHARINDEX('%', @Pattern);
            DECLARE @WildcardEnd INT = LEN(@Pattern) - CHARINDEX('%', REVERSE(@Pattern)) + 1;

            -- 提取匹配的文本
            IF @WildcardStart > 0 AND @WildcardEnd > @WildcardStart
            BEGIN
                SET @ExtractedValue = SUBSTRING(@Text, @StartPos + @WildcardStart, 
                                               PATINDEX(SUBSTRING(@Pattern, @WildcardEnd, LEN(@Pattern)), 
                                                        SUBSTRING(@Text, @StartPos + @WildcardStart, LEN(@Text))) - 1);
            END
            ELSE
            BEGIN
                -- 如果没有通配符，返回整个匹配
                SET @ExtractedValue = SUBSTRING(@Text, @StartPos, LEN(@Pattern));
            END

            -- 插入结果表
            INSERT INTO @Results (PatternName, ExtractedValue)
            VALUES (@PatternName, @ExtractedValue);
        END

        FETCH NEXT FROM pattern_cursor INTO @PatternID, @PatternName, @Pattern;
    END;

    CLOSE pattern_cursor;
    DEALLOCATE pattern_cursor;

    RETURN;
END
GO

-- 从文件提取数据存储过程（需要启用xp_cmdshell）
CREATE PROCEDURE ExtractFromFile
    @FilePath NVARCHAR(500),
    @OutputTable NVARCHAR(100)
AS
BEGIN
    DECLARE @Text NVARCHAR(MAX);

    -- 读取文件内容（注意：需要启用xp_cmdshell）
    -- 警告：启用xp_cmdshell可能带来安全风险
    EXEC sp_configure 'show advanced options', 1;
    RECONFIGURE;
    EXEC sp_configure 'xp_cmdshell', 1;
    RECONFIGURE;

    -- 读取文件内容到变量
    CREATE TABLE #FileContent(Content NVARCHAR(MAX));
    INSERT INTO #FileContent
    EXEC xp_cmdshell 'type ' + @FilePath;

    SELECT @Text = COALESCE(@Text + CHAR(13) + CHAR(10), '') + Content
    FROM #FileContent
    WHERE Content IS NOT NULL;

    DROP TABLE #FileContent;

    -- 提取数据并插入到输出表
    EXEC('INSERT INTO ' + @OutputTable + ' SELECT * FROM ExtractFromText(''' + REPLACE(@Text, '''', '''''') + ''')');

    -- 禁用xp_cmdshell
    EXEC sp_configure 'xp_cmdshell', 0;
    RECONFIGURE;
    EXEC sp_configure 'show advanced options', 0;
    RECONFIGURE;
END
GO

-- 保存结果到JSON文件存储过程
CREATE PROCEDURE SaveResultsToJson
    @InputTable NVARCHAR(100),
    @OutputFilePath NVARCHAR(500)
AS
BEGIN
    -- 使用FOR JSON AUTO生成JSON
    DECLARE @Json NVARCHAR(MAX);
    EXEC('SELECT @Json = (SELECT * FROM ' + @InputTable + ' FOR JSON AUTO)' , @Json OUTPUT);

    -- 写入文件（需要启用xp_cmdshell）
    EXEC sp_configure 'show advanced options', 1;
    RECONFIGURE;
    EXEC sp_configure 'xp_cmdshell', 1;
    RECONFIGURE;

    -- 将JSON写入文件
    DECLARE @Cmd NVARCHAR(1000);
    SET @Cmd = 'echo ' + REPLACE(@Json, '''', '''''') + ' > ' + @OutputFilePath;
    EXEC xp_cmdshell @Cmd;

    -- 禁用xp_cmdshell
    EXEC sp_configure 'xp_cmdshell', 0;
    RECONFIGURE;
    EXEC sp_configure 'show advanced options', 0;
    RECONFIGURE;
END
GO

-- 医疗数据提取示例
-- 添加医疗数据提取模式
-- 注意：SQL Server的PATINDEX不支持完整正则表达式，使用通配符近似匹配
EXEC AddRegexPattern
    @PatternName = 'patient_name',
    @Pattern = '%患者姓名[:：]%[吖-座]%',
    @Description = '提取患者姓名（使用中文字符范围近似匹配）';

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
    @Description = '提取诊断信息（使用中文字符范围近似匹配）';

EXEC AddRegexPattern
    @PatternName = 'treatment',
    @Pattern = '%治疗[:：]%[吖-座，。,;；]%',
    @Description = '提取治疗信息（使用中文字符范围近似匹配）';

EXEC AddRegexPattern
    @PatternName = 'medicine',
    @Pattern = '%药物[:：]%[吖-座]%[片胶囊注射液颗粒]%',
    @Description = '提取药物信息（使用中文字符范围近似匹配）';

EXEC AddRegexPattern
    @PatternName = 'date',
    @Pattern = '%[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]%',
    @Description = '提取日期（YYYY-MM-DD格式）';

-- 创建示例表并测试
CREATE TABLE #TestResults (
    PatternName NVARCHAR(100),
    ExtractedValue NVARCHAR(MAX)
);

-- 示例文本
DECLARE @SampleText NVARCHAR(MAX) = N'
患者信息：
患者姓名：张三
性别：男
年龄：45岁
就诊号：B000831020
诊断：高血压、糖尿病
治疗：口服药物治疗
药物：降压片、降糖胶囊
就诊日期：2024-05-15
';

-- 提取数据
INSERT INTO #TestResults
SELECT * FROM ExtractFromText(@SampleText);

-- 查询结果
SELECT * FROM #TestResults;

-- 清理临时表
DROP TABLE #TestResults;

-- 说明：
-- 1. 本脚本提供了一个基础框架，实际使用时可能需要根据具体需求调整
-- 2. SQL Server的PATINDEX函数支持有限的通配符，不支持完整的正则表达式
-- 3. 对于更复杂的正则表达式需求，可以考虑启用SQL Server的CLR集成
-- 4. 使用xp_cmdshell存在安全风险，生产环境中应谨慎使用