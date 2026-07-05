#include "oefp/column_request.h"

#include <gtest/gtest.h>

using namespace OEFP;

TEST(ColumnRequestTest, AllWantsEverything) {
    const auto req = ColumnRequest::All();
    EXPECT_TRUE(req.WantsAll());
    EXPECT_TRUE(req.Wants(0));
    EXPECT_TRUE(req.Wants(9999));
}

TEST(ColumnRequestTest, SubsetWantsOnlyListed) {
    const auto req = ColumnRequest::Subset({2, 5});
    EXPECT_FALSE(req.WantsAll());
    EXPECT_TRUE(req.Wants(2));
    EXPECT_TRUE(req.Wants(5));
    EXPECT_FALSE(req.Wants(3));
}

TEST(ColumnRequestTest, EmptySubsetWantsNothing) {
    const auto req = ColumnRequest::Subset({});
    EXPECT_FALSE(req.WantsAll());
    EXPECT_FALSE(req.Wants(0));
}
