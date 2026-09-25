// 模型 E2E 性能范围：SFPS 输出张量在设备内存就绪之后，到最终 score 输出。
// sfps_feature_embedding 保留为输入与引用声明；不建模其查表、表容量、通信或缓存。
local model_structure = [
  {
    name: 'all_sparse_input',
    type: 'sfps_feature_embedding',
    parameters: {
      methods: {
        // "type == 'discrete' and 'alltoall' not in region": {
        "type == 'discrete' and name == 'slot_id'": {
          communication_type: 'remote',
          filter_threshold: 1,
          shrink_step_threshold: 80000,
          feature_policy: 'counter_filter_with_default',
          key_hash_function: {
            "type": "mmhash2"
          },
          embedding_size: 24,
          optimizer: 'adam',
          learning_rate: 0.0001,
          // step_threshold: 20,
        },
        "type == 'discrete' and name in ['auid_clk_slotid_dts_ubd_list', 'auid_clk_slotid_crtv_ubd_list', 'auid_hwdsp_clk_slotid_dts_ubd_list', 'auid_hwdsp_cls_slotid_dts_ubd_list', 'auid_hwdsp_imp_slotid_dfq_top8_30d', 'auid_imp_no_clk_slotid_dts_ubd_list', 'auid_hwdsp_clk_taskid_tfidf_list_180d', 'rt_auid_noclick_slot_list_12h', 'rt_auid_noclick_slot_list_7d', 'rt_auid_noinstall_slot_list_cnt_24h', 'rt_auid_show_slot_list_12h']": {
          communication_type: 'remote',
          filter_threshold: 1,
          shrink_step_threshold: 80000,
          feature_policy: 'counter_filter_with_default',
          key_hash_function: {
            "type": "mmhash2"
          },
          embedding_size: 40,
          optimizer: 'adam',
          learning_rate: 0.0001,
          // step_threshold: 20,
        },
        "type == 'discrete' and name in ['auid__site_app_pkg','auid__crtv_clstid_stream','auid__site_id','auid__slot_id','auid__task_id','slot_id__crtv_clstid_stream','slot_id__task_id','site_app_pkg__crtv_clstid_stream','site_app_pkg__task_id','crtv_clstid_stream__site_id','task_id__site_id']": {
          communication_type: 'remote',
          filter_threshold: 5,
          shrink_step_threshold: 80000,
          feature_policy: 'counter_filter_with_default',
          key_hash_function: {
            "type": "mmhash2"
          },
          embedding_size: 16,
          optimizer: 'adam',
          learning_rate: 0.0001,
          // step_threshold: 20,
        },
        "type == 'discrete' and name not in ['slot_id', 'auid_clk_slotid_dts_ubd_list', 'auid_clk_slotid_crtv_ubd_list', 'auid_hwdsp_clk_slotid_dts_ubd_list', 'auid_hwdsp_cls_slotid_dts_ubd_list', 'auid_hwdsp_imp_slotid_dfq_top8_30d', 'auid_imp_no_clk_slotid_dts_ubd_list', 'auid_hwdsp_clk_taskid_tfidf_list_180d', 'rt_auid_noclick_slot_list_12h', 'rt_auid_noclick_slot_list_7d', 'rt_auid_noinstall_slot_list_cnt_24h', 'rt_auid_show_slot_list_12h','auid__site_app_pkg','auid__crtv_clstid_stream','auid__site_id','auid__slot_id','auid__task_id','slot_id__crtv_clstid_stream','slot_id__task_id','site_app_pkg__crtv_clstid_stream','site_app_pkg__task_id','crtv_clstid_stream__site_id','task_id__site_id']": {
          communication_type: 'remote',
          filter_threshold: 1,
          shrink_step_threshold: 80000,
          feature_policy: 'counter_filter_with_default',
          key_hash_function: {
            "type": "mmhash2"
          },
          embedding_size: 16,
          optimizer: 'adam',
          learning_rate: 0.0001,
          // step_threshold: 20,
        },
        sfps_emb_conf: {
          init_name: "truncate_norm",
          embedding_init_stddev: 0.02,
          pooling_type: "average",
          is_merge_feature_table: false,
          batch_size: 10240,
          hash_type: "hash",
        },
      },
      other_config: {
        table_info_path: "$data_dir/$scene_name/$model_id/model/$version/$date$hour/table_info.json"
      }
    },
    inputs: {
      inputs: "ref::inputs.features#discrete",
    },
    outputs: 'embedding',
  },

  {
    name: 'dense_input',
    type: 'log_dense',
    parameters: {
      embedding_size: 8,
      hash_num: 32,
      log_base: 2,
      rate_feature_list: [
        'auid_normalizedt2_ctr_90d',
        'auid_site_ctr_90d',
        'auid_normalizedt2_ctr_180d',
        'auid_advertiser_ctr_180d',
        'auid_appc3_ctr_180d',
        'auid_advertiser_ctr_90d',
        'auid_slot_ctr_180d',
        'auid_site_ctr_180d',
        'auid_slot_ctr_90d',
        'auid_ctr_90d',
        'auid_app_ctr_90d',
        'auid_ctr_180d',
        'rt_auid_ctr_7d',
        'auid_siteid_ctr_30d',
        'auid_slotid_ctr_7d',
        'slot_ctr_3rdpty_dsp_7d',
        'slot_ctr_3rdpty_dsp_30d',
        'auid_slotid_ctr_30d',
        'gender_slotid_ctr_today',
        'site_paid_rate_smooth_3d'
      ],
      embedding_initializer_config: {
        class_name: 'RandomUniform',
        config: {
          minval: -0.001,
          maxval: 0.001
        }
      },
      embedding_regularizer_config: {
        class_name: 'l1_l2',
        config: {
          l1: 0,
          l2: 0
        }
      }
    },
    inputs: {
      inputs: "ref::inputs.name != 'ssp_dsp_id_new' and name != 'rt_auid_appid_alldsp_imp_cnt_10minute' and type == 'continuous'",
    },
    outputs: 'embedding',
  },

  {
    name: 'postprocess',
    type: 'feature_sequential',
    parameters: {
      methods: {
        'features#multiple_discrete': [
          {
            type: 'sum',
            axis: 1
          }
        ],
        'features#single_discrete': [
          {
            type: "sum",
            axis: 1
          }
        ],
      },
    },
    inputs: {
      inputs: 'ref::all_sparse_input.embedding'
    },
    outputs: 'embedding',
  },

  {
    name: 'split_can_target_din',
    type: 'split',
    inputs: {
      value: 'ref::values(all_sparse_input.embedding.type == "discrete" and name in ["slot_id"])[0]',
      num_or_size_splits: [16, 8],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_din1',
    type: 'split',
    inputs: {
      value: 'ref::values(all_sparse_input.embedding.type == "discrete" and name == "auid_imp_no_clk_slotid_dts_ubd_list")[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_din2',
    type: 'split',
    inputs: {
      value: 'ref::values(all_sparse_input.embedding.type == "discrete" and name == "auid_clk_slotid_dts_ubd_list")[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_din3',
    type: 'split',
    inputs: {
      value: 'ref::values(all_sparse_input.embedding.type == "discrete" and name == "auid_clk_slotid_crtv_ubd_list")[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_din4',
    type: 'split',
    inputs: {
      value: 'ref::values(all_sparse_input.embedding.type == "discrete" and name == "auid_hwdsp_clk_slotid_dts_ubd_list")[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_din5',
    type: 'split',
    inputs: {
      value: 'ref::values(all_sparse_input.embedding.type == "discrete" and name == "auid_hwdsp_imp_slotid_dfq_top8_30d")[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'din',
    type: 'din',
    parameters: {
      din_specific_params: {
        target_feature: [
          "site_id",
          "creativetype",
          "hour_of_day",
          "hour_of_day",
          "hour_of_day",
          "advertiser_id"
        ],
        sequence_features: [
          ["auid_clk_siteid_dts_ubd_list"],
          ["rt_auid_show_creativetype_list_cnt_7d"],
          ["auid_clk_hour_crtv_ubd_list"],
          ["site_register_hour_top_30d"],
          ["site_paid_hour_top3_30d"],
          ["auid_clk_advrid_dts_ubd_list"]
        ],
        dnn_config: {
          type: 'dnn',
          hidden_dims: [32, 1],
          hidden_activation: ['dice']
        }
      },
      use_same_dnn: false
    },
    inputs: {
      inputs: 'ref::all_sparse_input.embedding.type == "discrete" and name in ["site_id","creativetype","hour_of_day","advertiser_id","auid_clk_siteid_dts_ubd_list","rt_auid_show_creativetype_list_cnt_7d","auid_clk_hour_crtv_ubd_list","site_register_hour_top_30d","site_paid_hour_top3_30d","auid_clk_advrid_dts_ubd_list"]'
    },
    outputs: "output"
  },

  {
    name: 'din1',
    type: 'din_v2',
    parameters: {
      din_specific_params: {
        target_feature: ["slot_id"],
        sequence_features: [
          ["auid_imp_no_clk_slotid_dts_ubd_list"]
        ],
        dnn_config: {
          type: 'dnn',
          hidden_dims: [32, 1],
          hidden_activation: ['dice']
        }
      },
      use_same_dnn: false
    },
    inputs: {
      inputs: ['ref::split_can_target_din.output[0]', 'ref::split_can_seq_din1.output[0]']
    },
    outputs: "output"
  },

  {
    name: 'din2',
    type: 'din_v2',
    parameters: {
      din_specific_params: {
        target_feature: ["slot_id"],
        sequence_features: [
          ["auid_clk_slotid_dts_ubd_list"]
        ],
        dnn_config: {
          type: 'dnn',
          hidden_dims: [32, 1],
          hidden_activation: ['dice']
        }
      },
      use_same_dnn: false
    },
    inputs: {
      inputs: ['ref::split_can_target_din.output[0]', 'ref::split_can_seq_din2.output[0]']
    },
    outputs: "output"
  },

  {
    name: 'din3',
    type: 'din_v2',
    parameters: {
      din_specific_params: {
        target_feature: ["slot_id"],
        sequence_features: [
          ["auid_clk_slotid_crtv_ubd_list"]
        ],
        dnn_config: {
          type: 'dnn',
          hidden_dims: [32, 1],
          hidden_activation: ['dice']
        }
      },
      use_same_dnn: false
    },
    inputs: {
      inputs: ['ref::split_can_target_din.output[0]', 'ref::split_can_seq_din3.output[0]']
    },
    outputs: "output"
  },

  {
    name: 'din4',
    type: 'din_v2',
    parameters: {
      din_specific_params: {
        target_feature: ["slot_id"],
        sequence_features: [
          ["auid_hwdsp_clk_slotid_dts_ubd_list"]
        ],
        dnn_config: {
          type: 'dnn',
          hidden_dims: [32, 1],
          hidden_activation: ['dice']
        }
      },
      use_same_dnn: false
    },
    inputs: {
      inputs: ['ref::split_can_target_din.output[0]', 'ref::split_can_seq_din4.output[0]']
    },
    outputs: "output"
  },

  {
    name: 'din5',
    type: 'din_v2',
    parameters: {
      din_specific_params: {
        target_feature: ["slot_id"],
        sequence_features: [
          ["auid_hwdsp_imp_slotid_dfq_top8_30d"]
        ],
        dnn_config: {
          type: 'dnn',
          hidden_dims: [32, 1],
          hidden_activation: ['dice']
        }
      },
      use_same_dnn: false
    },
    inputs: {
      inputs: ['ref::split_can_target_din.output[0]', 'ref::split_can_seq_din5.output[0]']
    },
    outputs: "output"
  },

  {
    name: 'din6',
    type: 'din',
    parameters: {
      din_specific_params: {
        target_feature: [
          "app_class2",
          "app_class2",
          "app_class2",
          "app_class2",
          "app_class3",
          "app_class3"
        ],
        sequence_features: [
          ["auid_query_tag2_dfq_top5_15d_list"],
          ["auid_query_tag2_dfq_top5_30d_list"],
          ["auid_query_tag2_dfq_top5_3d_list"],
          ["auid_query_tag2_dfq_top5_7d_list"],
          ["auid_query_tag3_dfq_top5_30d_list"],
          ["auid_query_tag3_dfq_top5_3d_list"]
        ],
        dnn_config: {
          type: 'dnn',
          hidden_dims: [32, 1],
          hidden_activation: ['dice']
        }
      },
      use_same_dnn: false
    },
    inputs: {
      inputs: 'ref::all_sparse_input.embedding.type == "discrete" and name in ["app_class2", "app_class3", "auid_query_tag2_dfq_top5_15d_list", "auid_query_tag2_dfq_top5_30d_list", "auid_query_tag2_dfq_top5_3d_list", "auid_query_tag2_dfq_top5_7d_list", "auid_query_tag3_dfq_top5_30d_list", "auid_query_tag3_dfq_top5_3d_list"]'
    },
    outputs: "output"
  },

  {
    name: 'din7',
    type: 'din',
    parameters: {
      din_specific_params: {
        target_feature: ["app_class2"],
        sequence_features: [
          ["auid_query_tag2_dfq_top10_30d_list"]
        ],
        dnn_config: {
          type: 'dnn',
          hidden_dims: [32, 1],
          hidden_activation: ['dice']
        }
      },
      use_same_dnn: false
    },
    inputs: {
      inputs: 'ref::all_sparse_input.embedding.type == "discrete" and name in ["app_class2", "auid_query_tag2_dfq_top10_30d_list"]'
    },
    outputs: "output"
  },

  {
    name: 'din8',
    type: 'din',
    parameters: {
      din_specific_params: {
        target_feature: ["app_class3"],
        sequence_features: [
          ["auid_query_tag3_dfq_top15_30d_list"]
        ],
        dnn_config: {
          type: 'dnn',
          hidden_dims: [32, 1],
          hidden_activation: ['dice']
        }
      },
      use_same_dnn: false
    },
    inputs: {
      inputs: 'ref::all_sparse_input.embedding.type == "discrete" and name in ["app_class3", "auid_query_tag3_dfq_top15_30d_list"]'
    },
    outputs: "output"
  },

  {
    name: 'flat_list_debias',
    type: 'flatten_list',
    inputs: {
      inputs: [
        'ref::values(postprocess.embedding.type == "discrete" and name in ["auid__site_app_pkg","auid__crtv_clstid_stream","auid__site_id","auid__slot_id","auid__task_id","slot_id__crtv_clstid_stream","slot_id__task_id","site_app_pkg__crtv_clstid_stream","site_app_pkg__task_id","crtv_clstid_stream__site_id","task_id__site_id","auid","crtv_clstid_stream","task_id","slot_id","site_id","site_app_pkg"])'
      ],
    },
    outputs: 'output',
  },

  {
    name: 'concat_debias',
    type: 'concatenate',
    inputs: {
      inputs: 'ref::flat_list_debias.output',
    },
    outputs: 'output',
  },

  {
    name: 'debias_tower',
    type: 'dnn',
    parameters: {
      hidden_dims: [64],
    },
    inputs: {
      inputs: 'ref::concat_debias.output'
    },
    outputs: 'output',
  },

  {
    name: 'split_can_target_feature',
    type: 'split',
    inputs: {
      value: 'ref::values(postprocess.embedding.type == "discrete" and name in ["slot_id"])[0]',
      num_or_size_splits: [16, 8],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_feature1',
    type: 'split',
    inputs: {
      value: 'ref::values(postprocess.embedding.type == "discrete" and name in ["auid_clk_slotid_dts_ubd_list"])[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_feature2',
    type: 'split',
    inputs: {
      value: 'ref::values(postprocess.embedding.type == "discrete" and name == "auid_clk_slotid_crtv_ubd_list")[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_feature3',
    type: 'split',
    inputs: {
      value: 'ref::values(postprocess.embedding.type == "discrete" and name == "auid_hwdsp_clk_slotid_dts_ubd_list")[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_feature4',
    type: 'split',
    inputs: {
      value: 'ref::values(postprocess.embedding.type == "discrete" and name == "auid_hwdsp_cls_slotid_dts_ubd_list")[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_feature5',
    type: 'split',
    inputs: {
      value: 'ref::values(postprocess.embedding.type == "discrete" and name == "auid_hwdsp_imp_slotid_dfq_top8_30d")[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_feature6',
    type: 'split',
    inputs: {
      value: 'ref::values(postprocess.embedding.type == "discrete" and name == "auid_imp_no_clk_slotid_dts_ubd_list")[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_feature7',
    type: 'split',
    inputs: {
      value: 'ref::values(postprocess.embedding.type == "discrete" and name == "auid_hwdsp_clk_taskid_tfidf_list_180d")[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_feature8',
    type: 'split',
    inputs: {
      value: 'ref::values(postprocess.embedding.type == "discrete" and name == "rt_auid_noclick_slot_list_12h")[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_feature9',
    type: 'split',
    inputs: {
      value: 'ref::values(postprocess.embedding.type == "discrete" and name == "rt_auid_noclick_slot_list_7d")[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_feature10',
    type: 'split',
    inputs: {
      value: 'ref::values(postprocess.embedding.type == "discrete" and name == "rt_auid_noinstall_slot_list_cnt_24h")[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'split_can_seq_feature11',
    type: 'split',
    inputs: {
      value: 'ref::values(postprocess.embedding.type == "discrete" and name == "rt_auid_show_slot_list_12h")[0]',
      num_or_size_splits: [16, 24],
      axis: -1,
    },
    outputs: 'output',
  },

  {
    name: 'epnet_domain',
    type: 'concatenate',
    inputs: {
      inputs: 'ref::values(postprocess.embedding.type == "discrete" and name in ["media_type_new","site_app_pkg"])',
    },
    outputs: 'output',
  },

  {
    name: 'dense_dict_concate_sence_slot',
    type: 'concatenate',
    inputs: {
      inputs: 'ref::values(dense_input.embedding[0]."sence_slot" in region)',
    },
    outputs: 'output',
  },

  {
    name: 'dense_squeeze_sence_slot',
    type: 'squeeze',
    inputs: {
      input: 'ref::dense_dict_concate_sence_slot.output',
      axis: 1
    },
    outputs: 'output',
  },

  {
    name: 'sence_slot',
    type: 'flatten_list',
    inputs: {
      inputs: [
        'ref::values(postprocess.embedding."sence_slot" in region)',
        'ref::dense_squeeze_sence_slot.output',
      ],
    },
    outputs: 'output',
  },

  {
    name: 'goods_prom',
    type: 'flatten_list',
    inputs: {
      inputs: ['ref::values(postprocess.embedding."goods_prom" in region)',],
    },
    outputs: 'output',
  },

  {
    name: 'dense_dict_concate_sence_media',
    type: 'concatenate',
    inputs: {
      inputs: 'ref::values(dense_input.embedding[0]."sence_media" in region)',
    },
    outputs: 'output',
  },

  {
    name: 'dense_squeeze_sence_media',
    type: 'squeeze',
    inputs: {
      input: 'ref::dense_dict_concate_sence_media.output',
      axis: 1
    },
    outputs: 'output',
  },

  {
    name: 'sence_media',
    type: 'flatten_list',
    inputs: {
      inputs: [
        'ref::values(postprocess.embedding."sence_media" in region)',
        'ref::dense_squeeze_sence_media.output',
      ],
    },
    outputs: 'output',
  },

  {
    name: 'dense_dict_concate_user_attribute',
    type: 'concatenate',
    inputs: {
      inputs: 'ref::values(dense_input.embedding[0]."user_attribute" in region)',
    },
    outputs: 'output',
  },

  {
    name: 'dense_squeeze_user_attribute',
    type: 'squeeze',
    inputs: {
      input: 'ref::dense_dict_concate_user_attribute.output',
      axis: 1
    },
    outputs: 'output',
  },

  {
    name: 'user_attribute',
    type: 'flatten_list',
    inputs: {
      inputs: [
        'ref::values(postprocess.embedding."user_attribute" in region)',
        'ref::dense_squeeze_user_attribute.output',
      ],
    },
    outputs: 'output',
  },

  {
    name: 'dense_dict_concate_user_ad_behavior_cross',
    type: 'concatenate',
    inputs: {
      inputs: 'ref::values(dense_input.embedding[0]."user_ad_behavior_cross" in region)',
    },
    outputs: 'output',
  },

  {
    name: 'dense_squeeze_user_ad_behavior_cross',
    type: 'squeeze',
    inputs: {
      input: 'ref::dense_dict_concate_user_ad_behavior_cross.output',
      axis: 1
    },
    outputs: 'output',
  },

  {
    name: 'user_ad_behavior_cross',
    type: 'flatten_list',
    inputs: {
      inputs: [
        'ref::values(postprocess.embedding."user_ad_behavior_cross" in region)',
        'ref::dense_squeeze_user_ad_behavior_cross.output',
      ],
    },
    outputs: 'output',
  },

  {
    name: 'sence_env',
    type: 'flatten_list',
    inputs: {
      inputs: ['ref::values(postprocess.embedding."sence_env" in region)',],
    },
    outputs: 'output',
  },

  {
    name: 'cross_feature',
    type: 'flatten_list',
    inputs: {
      inputs: [
        'ref::values(postprocess.embedding."cross_sence_sence" in region)',
        'ref::values(postprocess.embedding."cross_goods_sence" in region)',
        'ref::values(postprocess.embedding."cross_user_goods" in region)',
        'ref::values(postprocess.embedding."cross_user_sence" in region)',
      ],
    },
    outputs: 'output',
  },

  {
    name: 'user_app_behavior',
    type: 'flatten_list',
    inputs: {
      inputs: ['ref::values(postprocess.embedding."user_app_behavior" in region)',],
    },
    outputs: 'output',
  },

  {
    name: 'user_ad_behavior_goods',
    type: 'flatten_list',
    inputs: {
      inputs: ['ref::values(postprocess.embedding."user_ad_behavior_goods" in region)',],
    },
    outputs: 'output',
  },

  {
    name: 'user_ad_behavior_sence',
    type: 'flatten_list',
    inputs: {
      inputs: ['ref::values(postprocess.embedding."user_ad_behavior_sence" in region)',],
    },
    outputs: 'output',
  },

  {
    name: 'dense_dict_concate_user_ad_behavior',
    type: 'concatenate',
    inputs: {
      inputs: 'ref::values(dense_input.embedding[0]."user_ad_behavior" in region)',
    },
    outputs: 'output',
  },

  {
    name: 'dense_squeeze_user_ad_behavior',
    type: 'squeeze',
    inputs: {
      input: 'ref::dense_dict_concate_user_ad_behavior.output',
      axis: 1
    },
    outputs: 'output',
  },

  {
    name: 'user_ad_behavior',
    type: 'flatten_list',
    inputs: {
      inputs: [
        'ref::values(postprocess.embedding."user_ad_behavior" in region)',
        'ref::dense_squeeze_user_ad_behavior.output',
      ],
    },
    outputs: 'output',
  },

  {
    name: 'user_search_behavior',
    type: 'flatten_list',
    inputs: {
      inputs: ['ref::values(postprocess.embedding."user_search_behavior" in region)',],
    },
    outputs: 'output',
  },

  {
    name: 'din_group',
    type: 'flatten_list',
    inputs: {
      inputs: [
        'ref::din.output',
        'ref::din1.output',
        'ref::din2.output',
        'ref::din3.output',
        'ref::din4.output',
        'ref::din5.output',
        'ref::din6.output',
        'ref::din7.output',
        'ref::din8.output'
      ],
    },
    outputs: 'output',
  },

  {
    name: 'flat_list_all',
    type: 'flatten_list',
    inputs: {
      inputs: [
        'ref::sence_slot.output',
        'ref::goods_prom.output',
        'ref::sence_media.output',
        'ref::user_attribute.output',
        'ref::user_ad_behavior_cross.output',
        'ref::sence_env.output',
        'ref::cross_feature.output',
        'ref::user_app_behavior.output',
        'ref::user_ad_behavior_goods.output',
        'ref::user_ad_behavior_sence.output',
        'ref::user_ad_behavior.output',
        'ref::user_search_behavior.output',
        'ref::din_group.output'
      ],
    },
    outputs: 'output',
  },

  {
    name: 'can_network',
    type: 'ads_can',
    parameters: {
      can_sequence_embedding_size: 24,
      can_target_embedding_size: 8
    },
    inputs: {
      inputs: [
        [
          'ref::split_can_seq_feature1.output[1]',
          'ref::split_can_seq_feature2.output[1]',
          'ref::split_can_seq_feature3.output[1]',
          'ref::split_can_seq_feature4.output[1]',
          'ref::split_can_seq_feature5.output[1]',
          'ref::split_can_seq_feature6.output[1]',
          'ref::split_can_seq_feature7.output[1]',
          'ref::split_can_seq_feature8.output[1]',
          'ref::split_can_seq_feature9.output[1]',
          'ref::split_can_seq_feature10.output[1]',
          'ref::split_can_seq_feature11.output[1]'
        ],
        ['ref::split_can_target_feature.output[1]']
      ]
    },
    outputs: 'output',
  },

  {
    name: 'concat_all',
    type: 'concatenate',
    inputs: {
      inputs: 'ref::flat_list_all.output',
    },
    outputs: 'output',
  },

  {
    name: 'concat_can',
    type: 'concatenate',
    inputs: {
      inputs: ['ref::concat_all.output', 'ref::can_network.output'],
    },
    outputs: 'output',
  },

  {
    name: 'epnet_output',
    type: 'epnet',
    parameters: {
      domain_stop_grad: true
    },
    inputs: {
      inputs: ["ref::epnet_domain.output", "ref::concat_can.output"]
    },
    outputs: 'output',
  },

  {
    name: 'concat_main_gate',
    type: 'concatenate',
    inputs: {
      inputs: ['ref::postprocess.embedding.app_id', 'ref::epnet_output.output'],
    },
    outputs: 'output',
  },

  {
    name: 'embedding_gate',
    type: 'gatenu',
    parameters: {
      activation: "relu",
    },
    inputs: {
      inputs: ['ref::concat_main_gate.output', 'ref::epnet_output.output']
    },
    outputs: 'output',
  },

  {
    name: 'dense_all',
    type: 'dense',
    parameters: {
      units: 416
    },
    inputs: {
      inputs: 'ref::embedding_gate.output'
    },
    outputs: 'output',
  },

  {
    name: 'dense_layer',
    type: "loop",
    parameters: {
      loop: {
        type: "layer_seq",
        config_list: [
          {
            type: 'concatenate',
          },
          {
            type: 'dense',
            units: 416,
          }
        ]
      },
    },
    inputs: {
      inputs: [
        'ref::sence_slot.output',
        'ref::goods_prom.output',
        'ref::sence_media.output',
        'ref::user_attribute.output',
        'ref::user_ad_behavior_cross.output',
        'ref::sence_env.output',
        'ref::cross_feature.output',
        'ref::user_app_behavior.output',
        'ref::user_ad_behavior_goods.output',
        'ref::user_ad_behavior_sence.output',
        'ref::user_ad_behavior.output',
        'ref::user_search_behavior.output'
      ],
    },
    outputs: 'output',
  },

  {
    name: 'stack_group',
    type: 'stack',
    inputs: {
      values: [
        "ref::dense_all.output",
        "ref::dense_layer.output[0]",
        "ref::dense_layer.output[1]",
        "ref::dense_layer.output[2]",
        "ref::dense_layer.output[3]",
        "ref::dense_layer.output[4]",
        "ref::dense_layer.output[5]",
        "ref::dense_layer.output[6]",
        "ref::dense_layer.output[7]",
        "ref::dense_layer.output[8]",
        "ref::dense_layer.output[9]",
        "ref::dense_layer.output[10]",
        "ref::dense_layer.output[11]"
      ],
      axis: 1
    },
    outputs: 'output',
  },

  {
    name: 'token_mixer1',
    type: 'token_mixer',
    parameters: {
      num_tokens: 13,
      d_model: 416,
    },
    inputs: {
      inputs: 'ref::stack_group.output',
    },
    outputs: 'output',
  },

  {
    name: 'pffn1',
    type: 'per_token_ffn',
    // type: 'shared_ffn_lora',
    parameters: {
      num_tokens: 13,
      d_model: 416,
      dropout: 0.5,
      use_residual: true,
      use_norm: true
    },
    inputs: {
      inputs: 'ref::token_mixer1.output',
    },
    outputs: 'output',
  },

  {
    name: 'token_mixer2',
    type: 'token_mixer',
    parameters: {
      num_tokens: 13,
      d_model: 416,
    },
    inputs: {
      inputs: 'ref::pffn1.output',
    },
    outputs: 'output',
  },

  {
    name: 'pffn2',
    type: 'per_token_ffn',
    // type: 'shared_ffn_lora',
    parameters: {
      num_tokens: 13,
      d_model: 416,
      dropout: 0.5,
      use_residual: true,
      use_norm: true
    },
    inputs: {
      inputs: 'ref::token_mixer2.output',
    },
    outputs: 'output',
  },

  {
    name: 'tensor_sum',
    type: 'sum',
    parameters: {
      axis: 1
    },
    inputs: {
      inputs: 'ref::pffn2.output',
    },
    outputs: 'output',
  },

  {
    name: 'densenet_1',
    type: 'dense',
    parameters: {
      units: 256,
      activation: 'relu',
      kernel_initializer: 'glorot_uniform',
      bias_initializer: 'zero',
    },
    inputs: {
      inputs: 'ref::embedding_gate.output'
    },
    outputs: 'output',
  },

  {
    name: 'gate1',
    type: 'gatenu',
    parameters: {
      activation: "relu",
    },
    inputs: {
      inputs: ['ref::concat_main_gate.output', 'ref::densenet_1.output']
    },
    outputs: 'output',
  },

  {
    name: 'densenet_2',
    type: 'dense',
    parameters: {
      units: 64,
      activation: 'relu',
      kernel_initializer: 'glorot_uniform',
      bias_initializer: 'zero',
    },
    inputs: {
      inputs: 'ref::gate1.output'
    },
    outputs: 'output',
  },

  {
    name: 'gate2',
    type: 'gatenu',
    parameters: {
      activation: "relu",
    },
    inputs: {
      inputs: ['ref::concat_main_gate.output', 'ref::densenet_2.output']
    },
    outputs: 'output',
  },

  {
    name: 'densenet_3',
    type: 'dense',
    parameters: {
      units: 128,
      activation: 'relu',
      kernel_initializer: 'glorot_uniform',
      bias_initializer: 'zero',
    },
    inputs: {
      inputs: 'ref::gate2.output'
    },
    outputs: 'output',
  },

  {
    name: 'gate3',
    type: 'gatenu',
    parameters: {
      activation: "relu",
    },
    inputs: {
      inputs: ['ref::concat_main_gate.output', 'ref::densenet_3.output']
    },
    outputs: 'output',
  },

  {
    name: 'concat_output',
    type: 'concatenate',
    inputs: {
      inputs: ['ref::gate3.output', 'ref::tensor_sum.output']
    },
    outputs: 'output',
  },

  {
    name: 'hr_tower',
    type: 'hr_tower',
    parameters: {
      hr_layer_dim: 12
    },
    inputs: {
      inputs: 'ref::concat_output.output',
      domain_feature_list: 'ref::values(postprocess.embedding.type == "discrete" and name in ["slot_id", "site_app_pkg", "media_type_new"])'
    },
    outputs: 'output',
  },

  {
    name: 'add_debias',
    type: 'concatenate',
    inputs: {
      inputs: ['ref::debias_tower.output', 'ref::hr_tower.output']
    },
    outputs: 'logit',
  },

  {
    name: 'linear',
    type: 'dense',
    parameters: {
      units: 1,
      activation: 'sigmoid',
    },
    inputs: {
      inputs: 'ref::add_debias.logit'
    },
    outputs: 'logit',
  },

  // negative_sampling校准,只影响打分，不影响train
  {
    name: 'negative_sampling',
    type: 'string_expression',
    parameters: {
      expression: 'x/(x+(1-x)/y)'
    },
    inputs: {
      inputs: {
        x: 'ref::linear.logit',
        y: 0.1
      }
    },
    outputs: 'score',
  },
];

model_structure
