//
// Created by 王奥博 on 2023/8/29.
//

#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

/**
+ - - - - - - - - - - - - - - - - - - - -  +
' Pass:                                    '
'                                          '
' +---------+     +--------------------+   '     +----------------+
' |PermuteOp| --> |   Depth2SpaceOp    |   ' --> |  Depth2SpaceOp |
' +---------+     +--------------------+   '     +----------------+
'                                          '
+ - - - - - - - - - - - - - - - - - - - - -+
*/

struct Depth2SpaceWithPermuteOpt : public OpRewritePattern<Depth2SpaceOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(Depth2SpaceOp op, PatternRewriter &rewriter) const override {
        return failure();
    }
};

void Depth2SpaceOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
    results.insert<Depth2SpaceWithPermuteOpt>(context);
}