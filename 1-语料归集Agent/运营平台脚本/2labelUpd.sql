INSERT INTO corpus_label (id, name, e_name, level, num, create_name, create_time, update_name, update_time, p_id, spt1, spt2, spt3, spt4, spt5) VALUES ('001005', '音频', 'Modal::Audio', 2, 4, NULL, '2025-01-16 00:00:00', NULL, '2025-01-16 00:00:00', '001', NULL, NULL, NULL, NULL, NULL);

INSERT INTO corpus_label (id, name, e_name, level, num, create_name, create_time, update_name, update_time, p_id, spt1, spt2, spt3, spt4, spt5) VALUES ('001004', '图片', 'Modal::Image', 2, 3, NULL, '2025-01-16 00:00:00', NULL, '2025-01-16 00:00:00', '001', NULL, NULL, NULL, NULL, NULL);

INSERT INTO corpus_label (id, name, e_name, level, num, create_name, create_time, update_name, update_time, p_id, spt1, spt2, spt3, spt4, spt5) VALUES ('001002', '多模态', 'Modal::Multimodal', 2, 1, NULL, '2025-01-16 00:00:00', NULL, '2025-01-16 00:00:00', '001', NULL, NULL, NULL, NULL, NULL);

INSERT INTO corpus_label (id, name, e_name, level, num, create_name, create_time, update_name, update_time, p_id, spt1, spt2, spt3, spt4, spt5) VALUES ('003002', '中文', 'Language::Chinese', 2, 2, NULL, '2025-01-16 00:00:00', NULL, '2025-01-16 00:00:00', '003', NULL, NULL, NULL, NULL, NULL);

INSERT INTO corpus_label (id, name, e_name, level, num, create_name, create_time, update_name, update_time, p_id, spt1, spt2, spt3, spt4, spt5) VALUES ('006002', '加工生产成品', 'c-source::product', 2, 2, '1', '2024-04-23 16:15:24', NULL, NULL, '006', NULL, NULL, NULL, NULL, NULL);


delete FROM corpus_label WHERE id like '003%' AND LENGTH(id) > 3 AND id NOT IN ('003002','003003');

INSERT INTO inbound_list_label (id, name, e_name, level, num, create_name, create_time, update_name, update_time, p_id) VALUES ('001005', '音频', 'Modal::Audio', 2, 4, NULL, '2025-01-16 00:00:00', NULL, '2025-01-16 00:00:00', '001');

INSERT INTO inbound_list_label (id, name, e_name, level, num, create_name, create_time, update_name, update_time, p_id) VALUES ('001004', '图片', 'Modal::Image', 2, 3, NULL, '2025-01-16 00:00:00', NULL, '2025-01-16 00:00:00', '001');

INSERT INTO inbound_list_label (id, name, e_name, level, num, create_name, create_time, update_name, update_time, p_id) VALUES ('001002', '多模态', 'Modal::Multimodal', 2, 1, NULL, '2025-01-16 00:00:00', NULL, '2025-01-16 00:00:00', '001');

INSERT INTO inbound_list_label (id, name, e_name, level, num, create_name, create_time, update_name, update_time, p_id) VALUES ('003002', '中文', 'Language::Chinese', 2, 2, NULL, '2025-01-16 00:00:00', NULL, '2025-01-16 00:00:00', '003');



delete FROM inbound_list_label WHERE id like '003%' AND LENGTH(id) > 3 AND id NOT IN ('003002','003003');