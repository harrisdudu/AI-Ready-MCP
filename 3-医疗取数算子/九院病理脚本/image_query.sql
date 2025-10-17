-- 图片查询SQL语句
-- 将原有的报告查询逻辑转换为图片查询逻辑

SELECT 
    main.VisitNumber AS 流水号,
    main.ReportDate AS 审核时间,
    main.FillerOrderNo AS 检查单号,
    main.ClassDescription AS 检查名称,
    sub.Description AS 检查所见,
    NULL AS 检查结果, -- 图片表中没有直接对应的检查结果字段
    main.ClassDescription AS 检查部位
FROM 
    CDR_OBS.dbo.OBS_ReportImageList main
INNER JOIN 
    CDR_OBS.dbo.OBS_ReportImageListSub sub 
    ON main.FillerOrderNo = sub.FillerOrderNo 
    AND main.ClassID = sub.ClassID 
    AND main.ImageNo = sub.ImageNo
WHERE 
    (main.ClassDescription LIKE '%内镜%' -- 对应原查询中的'NJ'分类
     OR main.ClassDescription LIKE '%病理%' -- 对应原查询中的'XHBL'分类
     OR (main.ClassDescription LIKE '%报告%' AND (sub.Description LIKE '%肠%' OR sub.Description LIKE '%胃%'))) -- 对应原查询中的'BL'分类
    AND main.VisitNumber IN (
        SELECT 
            VisitNumber
        FROM 
            CUSTUME.中间临时表2
    )
    AND main.IsDeleted = 0 -- 过滤已删除的记录
    AND sub.IsDeleted = 0 -- 过滤已删除的记录
ORDER BY 
    main.VisitNumber, main.ReportDate;

-- 说明：
-- 1. 使用OBS_ReportImageList作为主表，存储图片相关的基本信息
-- 2. 关联OBS_ReportImageListSub获取图片的详细描述信息
-- 3. 保留了原查询中的住院号过滤条件
-- 4. 由于图片表结构与原报告表不同，部分字段映射进行了调整
-- 5. 添加了IsDeleted=0的过滤条件以排除已删除的记录