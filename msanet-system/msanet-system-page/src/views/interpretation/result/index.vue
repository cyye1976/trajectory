<template>
  <div class="app-container">
    <el-table
      v-loading="listLoading"
      :data="list"
      element-loading-text="Loading"
      border
      fit
      highlight-current-row
    >
      <el-table-column align="center" label="ID" width="95">
        <template slot-scope="scope">
          {{ scope.row.id }}
        </template>
      </el-table-column>
      <el-table-column label="模型号" width="95">
        <template slot-scope="scope">
          {{ scope.row.modelId }}
        </template>
      </el-table-column>
      <el-table-column label="数据号" width="95">
        <template slot-scope="scope">
          {{ scope.row.dataId }}
        </template>
      </el-table-column>
      <el-table-column label="文件路径" align="center">
        <template slot-scope="scope">
          {{ scope.row.jsonUrl }}
        </template>
      </el-table-column>
      <el-table-column align="center" label="创建时间" width="200">
        <template slot-scope="scope">
          <i class="el-icon-time" />
          <span>{{ scope.row.createTime }}</span>
        </template>
      </el-table-column>
      <el-table-column align="center" label="操作" width="200">
        <template slot-scope="scope">
          <el-button type="primary" icon="el-icon-s-data" @click="visClick(scope.row.id)"></el-button>
          <el-button type="danger" icon="el-icon-delete" @click="deleteOnClick(scope.row.id)"></el-button>
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script>

import {getAction, postAction} from "@/api/manage";

export default {
  data() {
    return {
      list: null,
      listLoading: true,
      apiUrl:{
        list: '/interpretation/list',
        delete: '/interpretation/delete'
      }
    }
  },
  created() {
    this.fetchData()
  },
  methods: {
    fetchData() {
      this.listLoading = true
      getAction(this.apiUrl.list).then((res) => {
        console.log(res.data.result.records)
        this.list = res.data.result.records
        this.listLoading = false
      })
    },
    deleteOnClick(e){
      postAction(this.apiUrl.delete + "/" + e).then((res)=>{
        console.log(res.data)
        this.fetchData()
      })
    },
    visClick(e){
      this.$router.push({path:'/vis/vis', query:{id: e}})
    }
  }
}
</script>
