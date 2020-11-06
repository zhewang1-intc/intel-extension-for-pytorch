#ifndef DIL_OPERATORS_VANILLA_RNN_HPP
#define DIL_OPERATORS_VANILLA_RNN_HPP

namespace dil {

struct rnn_forward : public dnnl::vanilla_rnn_forward {

  using super = dnnl::vanilla_rnn_forward;

  static void compute(const dims& output_sizes,
                      const tensor& src_layer,
                      const tensor& src_iter,
                      const tensor& weights_layer,
                      const tensor& weights_iter,
                      const tensor& bias,
                      tensor& dst_layer,
                      tensor& dst_iter,
                      const rnn_kind akind,
                      const bool reverse = false,
                      prop_kind aprop = prop_kind::forward,
                      const engine& aengine = engine::cpu_engine()) {

    bool with_workspace = aprop == prop_kind::forward_training;
    auto activation = utils::rnn_kind_to_activation(akind);
    auto direction = reverse ? rnn_direction::unidirectional_right2left
                             : rnn_direction::unidirectional_left2right;
    auto src_layer_desc = src_layer.get_desc();
    auto src_iter_desc = src_iter.get_desc();
    // use any format for weights
    auto weights_layer_desc = weights_layer.get_desc().to_format_any();
    auto weights_iter_desc = weights_iter.get_desc().to_format_any();
    auto bias_desc = bias.get_desc();
    tensor::desc dst_layer_desc(output_sizes, src_layer.get_data_type(), tag::tnc);
    auto dst_iter_desc = dst_iter.get_desc();

    auto pd = primitive_desc(
        {aprop, activation, direction, src_layer_desc, src_iter_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, dst_iter_desc},
        aengine);

    auto expected_src_layer = src_layer.reorder_if_differ_in(pd.src_layer_desc());
    auto expected_src_iter = src_iter.reorder_if_differ_in(pd.src_iter_desc());

    auto expected_weights_layer = weights_layer.reorder_if_differ_in(pd.weights_desc());
    auto expected_weights_iter = weights_iter.reorder_if_differ_in(pd.weights_iter_desc());
    auto expected_bias = bias.reorder_if_differ_in(pd.bias_desc());

    dst_layer.reinit_if_possible(pd.dst_layer_desc());


    exec_args args {{DNNL_ARG_SRC_LAYER, expected_src_layer},
                    {DNNL_ARG_SRC_ITER, expected_src_iter},
                    {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
                    {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
                    {DNNL_ARG_BIAS, expected_bias},
                    {DNNL_ARG_DST_LAYER, dst_layer},
                    {DNNL_ARG_DST_ITER, dst_iter}};

    if (with_workspace) {
      dst_layer.init_workspace(pd.workspace_desc());
      args.insert({DNNL_ARG_WORKSPACE, dst_layer.get_workspace()});
    }

    super(pd).execute(stream::default_stream(), args);
  }
};

struct rnn_backward : public dnnl::vanilla_rnn_backward {

  using super = dnnl::vanilla_rnn_backward;

  static void compute(const tensor& src_layer,
                      const tensor& src_iter,
                      const tensor& weights_layer,
                      const tensor& weights_iter,
                      const tensor& bias,
                      const tensor& dst_layer,
                      const tensor& dst_iter,
                      const tensor& diff_dst_layer,
                      const tensor& diff_dst_iter,
                      tensor& diff_src_layer,
                      tensor& diff_src_iter,
                      tensor& diff_weights_layer,
                      tensor& diff_weights_iter,
                      tensor& diff_bias,
                      const rnn_kind akind,
                      const bool reverse = false,
                      const engine& aengine = engine::cpu_engine()) {
    auto aprop = prop_kind::backward;
    auto activation = utils::rnn_kind_to_activation(akind);
    auto direction = reverse ? rnn_direction::unidirectional_right2left
                             : rnn_direction::unidirectional_left2right;

    auto src_layer_desc = src_layer.get_desc();
    auto src_iter_desc = src_iter.get_desc();
    // use any format for weights
    auto weights_layer_desc = weights_layer.get_desc().to_format_any();
    auto weights_iter_desc = weights_iter.get_desc().to_format_any();
    auto bias_desc = bias.get_desc();
    auto dst_layer_desc = dst_layer.get_desc();
    auto dst_iter_desc = dst_iter.get_desc();
    auto diff_src_layer_desc = diff_src_layer.get_desc();
    auto diff_src_iter_desc = diff_src_iter.get_desc();
    auto diff_weights_layer_desc = diff_weights_layer.get_desc();
    auto diff_weights_iter_desc = diff_weights_iter.get_desc();
    auto diff_bias_desc = diff_bias.get_desc();
    auto diff_dst_layer_desc = diff_dst_layer.get_desc();
    auto diff_dst_iter_desc = diff_dst_iter.get_desc();

    auto forward_hints =
        dnnl::vanilla_rnn_forward::primitive_desc(
            {prop_kind::forward_training, activation, direction, src_layer_desc, src_iter_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, dst_iter_desc},
        aengine);

    auto pd = primitive_desc(
        {aprop, activation, direction, src_layer_desc, src_iter_desc,
         weights_layer_desc, weights_iter_desc, bias_desc,
         dst_layer_desc, dst_iter_desc,
         diff_src_layer_desc, diff_src_iter_desc,
         diff_weights_layer_desc, diff_weights_iter_desc, diff_bias_desc,
         diff_dst_layer_desc, diff_dst_iter_desc},
        aengine, forward_hints);
    
    auto expected_weights_layer = weights_layer.reorder_if_differ_in(pd.weights_desc());
    auto expected_weights_iter = weights_iter.reorder_if_differ_in(pd.weights_iter_desc());

    auto expected_workspace = dst_layer.get_workspace().reorder_if_differ_in(pd.workspace_desc());

    super(pd).execute(stream::default_stream(),
                      {{DNNL_ARG_SRC_LAYER, src_layer},
                       {DNNL_ARG_SRC_ITER, src_iter},
                       {DNNL_ARG_WEIGHTS_LAYER, expected_weights_layer},
                       {DNNL_ARG_WEIGHTS_ITER, expected_weights_iter},
                       {DNNL_ARG_BIAS, bias},
                       {DNNL_ARG_DST_LAYER, dst_layer},
                       {DNNL_ARG_DST_ITER, dst_iter},
                       {DNNL_ARG_DIFF_SRC_LAYER, diff_src_layer},
                       {DNNL_ARG_DIFF_SRC_ITER, diff_src_iter},
                       {DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_weights_layer},
                       {DNNL_ARG_DIFF_WEIGHTS_ITER, diff_weights_iter},
                       {DNNL_ARG_DIFF_BIAS, diff_bias},
                       {DNNL_ARG_DIFF_DST_LAYER, diff_dst_layer},
                       {DNNL_ARG_DIFF_DST_ITER, diff_dst_iter},
                       {DNNL_ARG_WORKSPACE, expected_workspace}});
  }
};

}  // namespace dil

#endif
