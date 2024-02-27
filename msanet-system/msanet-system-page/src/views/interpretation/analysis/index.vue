<template>
  <div class="app-container">
    <el-steps :active="active" finish-status="success">
      <el-step title="选择数据" ></el-step>
      <el-step title="选择模型" ></el-step>
      <el-step title="分析结果" ></el-step>
    </el-steps>
    <div style="padding-top: 40px;text-align: center">
      <div v-if="dataVisible">
        <div>
          <el-table
            v-loading="listLoading"
            :data="list"
            element-loading-text="Loading"
            border
            fit
            highlight-current-row
            @current-change="handleCurrentChange">
            <el-table-column align="center" label="ID" width="95">
              <template slot-scope="scope">
                {{ scope.row.id }}
              </template>
            </el-table-column>
            <el-table-column label="描述" align="center">
              <template slot-scope="scope">
                {{ scope.row.description }}
              </template>
            </el-table-column>
            <el-table-column align="center" label="创建时间" width="200">
              <template slot-scope="scope">
                <i class="el-icon-time" />
                <span>{{ scope.row.createTime }}</span>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
      <div v-if="modelVisible">
        <el-table
          v-loading="listLoading"
          :data="list"
          element-loading-text="Loading"
          border
          fit
          highlight-current-row
          @current-change="handleCurrentChange">
            <el-table-column align="center" label="ID" width="95">
              <template slot-scope="scope">
                {{ scope.row.id}}
              </template>
            </el-table-column>
            <el-table-column label="名称">
              <template slot-scope="scope">
                {{ scope.row.name }}
              </template>
            </el-table-column>
            <el-table-column label="描述" align="center">
              <template slot-scope="scope">
                <span>{{ scope.row.description }}</span>
              </template>
            </el-table-column>
            <el-table-column align="center" prop="created_at" label="创建时间" width="200">
              <template slot-scope="scope">
                <i class="el-icon-time" />
                <span>{{ scope.row.createTime }}</span>
              </template>
            </el-table-column>
        </el-table>
      </div>
      <div v-if="active === 0">
        <el-button style="margin-top: 12px;" class="el-footer" type="success" @click="next">下一步</el-button>
      </div>
      <div v-if="active === 1">
        <el-button style="margin-top: 12px;" class="el-footer" type="danger" @click="back">返回</el-button>
        <el-button style="margin-top: 12px;" class="el-footer" type="success" @click="next">下一步</el-button>
      </div>
      <div v-if="loadResult === true">
        <el-main v-loading="loadResult" v-if="loadResult === true" style="height: 200px;">
            正在分析数据，请稍后...
        </el-main>
<!--        <el-button type="primary" @click="testClick" v-if="loadResult == true">完成</el-button>-->
      </div>
      <div v-if="active === 3">
        <div v-if="loadResult === false">
            <span>
            <el-image style="width: 100px; height: 100px" :src="url"></el-image>
            </span><br>
          <div style="height: 30px"></div>
          <span>分析完成</span><br>
          <div style="height: 10px"></div>
          <span>请根据以下按钮提示进行操作</span>
        </div>
        <div style="padding-top: 30px" v-if="loadResult === false">
          <span style="padding-right: 20px">
            <el-button type="danger" @click="handleReset">返回第一步</el-button>
          </span>
          <el-button type="primary" @click="visClick(resultId)">可视化结果</el-button>
        </div>
      </div>
    </div>



  </div>
</template>

<script>

import {getAction, postAction} from "@/api/manage";

export default {
  data() {
    return {
      active: 0,
      listLoading: false,
      list:null,
      dataBean:null,
      modelBean:null,
      dataVisible: true,
      modelVisible: false,
      loadResult:false,
      resultId: 0,
      apiUrl:{
        getDataList: '/data/list',
        getModelList: '/model/list',
        analyseData: '/interpretation/analyseData'
      },
      url: "https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fpic.qiantucdn.com%2F58pic%2F27%2F57%2F83%2F13s58PIC2RK_1024.jpg%21%2Ffw%2F1024%2Fwatermark%2Furl%2FL2ltYWdlcy93YXRlcm1hcmsvZGF0dS5wbmc%3D%2Frepeat%2Ftrue%2Fcrop%2F0x1009a0a0&refer=http%3A%2F%2Fpic.qiantucdn.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1644507104&t=ae53677c44bca2640b90c29c1c178cb8"
    }
  },
  created() {
    this.fetchDataList()
  },
  methods: {
    next() {
      console.log(this.active)
      if (this.active === 0){
        this.dataVisible = false

        this.fetchModelList()
        this.modelVisible = true
      }
      if (this.active === 1){
        this.modelVisible = false
        this.loadResult = true
        // 执行数据分析逻辑
        postAction(this.apiUrl.analyseData + "/" + this.dataBean.id + "/" + this.modelBean.id).then((res)=>{
          this.timer = setTimeout(()=>{   //TODO:设置延迟执行,如果模型算法库可用则记得去掉
            console.log(res.data.result)
            this.resultId = res.data.result
            this.loadResult = false
            this.active++
          },3000);
        })
      }

      this.active++
    },
    back(){
      if (this.active === 1){
        this.modelVisible = false
        this.dataVisible = true
      }
      this.active--
    },
    fetchDataList(){
      this.listLoading = true
      getAction(this.apiUrl.getDataList).then((res)=>{
        console.log(res.data.result.records)
        this.list = res.data.result.records
        this.listLoading = false
      })
    },
    fetchModelList(){
      this.listLoading = true
      getAction(this.apiUrl.getModelList).then((res)=>{
        console.log(res.data.result.records)
        this.list = res.data.result.records
        this.listLoading = false
      })
    },
    handleCurrentChange(val){
      if (this.active === 0){
        this.dataBean = val
      }
      if (this.active === 1){
        this.modelBean = val
      }
    },
    handleReset(){
      this.active = 0
      this.dataVisible = true
      this.dataBean = null
      this.modelBean = null
    },
    testClick(){
      this.loadResult = false
      this.active++
    },
    visClick(e){
      console.log(e)
      this.$router.push({path:'/vis/vis', query:{id: e}})
    }
  }
}
</script>
