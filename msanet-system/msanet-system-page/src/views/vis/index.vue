<template>
  <div class="app-container">
      <el-steps :active="active" finish-status="success">
        <el-step title="完成相应操作"></el-step>
        <el-step title="可视化结果"></el-step>
      </el-steps>

    <div v-if="visMessageVisible" >
      <div style="padding-top: 30px;justify-content: center; display: flex;">
        <el-card :body-style="{ padding: '0px' }" shadow="hover">
          <el-image
            style="width: 400px; height: 200px"
            :src="this.visMessage.dataImageUrl"
            :preview-src-list="[this.visMessage.dataImageUrl]">
          </el-image>
          <div style="padding: 10px;">
            <el-upload
              ref="upload"
              class="upload-demo"
              :action="action"
              :limit="1"
              :data="formData"
              :on-change="onChangeUpload"
              :on-success="onSuccessUpload"
              :auto-upload="false">
              <el-button slot="trigger" size="small" type="primary">选取文件</el-button>
              <!--              <el-button style="margin-left: 10px;" size="small" type="success" @click="submitUpload">上传到服务器</el-button>-->
              <div slot="tip" class="el-upload__tip">只能上传jpg,png,bmp文件，且不超过10MB</div>
            </el-upload>
          </div>

          <div style="padding: 10px;">
            <div>要使用的模型: </div>
            <el-select label="要使用的模型" v-model="selectedModel" placeholder="请选择">
              <el-option
                v-for="item in modelOptions"
                :key="item.id"
                :label="item.name"
                :value="item.id">
              </el-option>
            </el-select>
          </div>
        </el-card>
      </div>
      <div style="padding-top: 30px;justify-content: center; display: flex;">
        <el-card :body-style="{ padding: '0px' }" shadow="hover">
          <el-transfer v-model="selectedVisContent" :data="visContent" :titles="['未选择的', '已选择的']"></el-transfer>
        </el-card>
      </div>
    </div>
    <div v-if="loadResult === true" style="text-align: center">
      <el-main v-loading="loadResult" v-if="loadResult === true" style="height: 200px;">
        正在结果可视化处理，请稍后...
      </el-main>
     </div>
    <div style="padding-top: 30px;justify-content: center; display: flex;" v-if="visMessageVisible">
      <el-card :body-style="{ padding: '0px' }" shadow="hover">
        <el-button class="el-footer" type="success" @click="next">下一步</el-button>
      </el-card>
    </div>
    <div v-if="active === 2" style="padding-top: 30px; text-align: center">
      <el-main>
        <el-row type="flex" justify="center" align="middle">
          <el-card  shadow="hover">

            <el-image
              style="width: 512px;height: 512px"
              :src="this.inferImage"
              :preview-src-list="[this.inferImage]">
            </el-image>
          </el-card>
          <el-card  shadow="hover">
            <div style="width: 450px; height: 512px">
              <span>详细信息</span>
              <div style="padding-bottom: 10px"></div>
<!--              <div-->
<!--                v-for="item in this.dataResultsList"-->
<!--                :style="`color: ${item.dataResultColors}`"-->
<!--              >-->
<!--              </div>-->
              <el-table
                :data="this.dataResultsList"
                border
                style="width: 100%;">
                  <el-table-column align="center" label="ID" width="40px">
                    <template slot-scope="scope">
                      <div :style="`color: ${scope.row.dataResultColors}`">
                      {{ scope.$index }}
                      </div>
                    </template>
                  </el-table-column>

                <el-table-column label="目标位置[X, Y]" >
                  <template slot-scope="scope">
                    <div :style="`color: ${scope.row.dataResultColors}`">
                      {{ scope.row.dataResults.centerXY }}
                    </div>
                  </template>
                </el-table-column>
                <el-table-column label="目标大小[W, H]" >
                  <template slot-scope="scope">
                    <div :style="`color: ${scope.row.dataResultColors}`">
                      {{ scope.row.dataResults.WH }}
                    </div>
                  </template>
                </el-table-column>
                <el-table-column label="目标航向角" >
                  <template slot-scope="scope">
                    <div :style="`color: ${scope.row.dataResultColors}`">
                      {{ scope.row.dataResults.course }}
                    </div>
                  </template>
                </el-table-column>
              </el-table>
            </div>
          </el-card>
        </el-row>
      </el-main>


      <div style="padding-top: 30px;justify-content: center; display: flex;">
        <el-card :body-style="{ padding: '0px' }" shadow="hover">
          <el-button type="primary" @click="">导出结果</el-button>
        </el-card>
      </div>
    </div>


  </div>
</template>

<style>
.avatar-uploader .el-upload {
  border: 1px dashed #d9d9d9;
  border-radius: 6px;
  cursor: pointer;
  position: relative;
  overflow: hidden;
}
.avatar-uploader .el-upload:hover {
  border-color: #409EFF;
}
.avatar-uploader-icon {
  font-size: 28px;
  color: #8c939d;
  width: 178px;
  height: 178px;
  line-height: 178px;
  text-align: center;
}
.avatar {
  width: 178px;
  height: 178px;
  display: block;
}


</style>

<script>

import { getAction, getActionBlob, postAction, postActionBlob } from '@/api/manage'
import { getApiBaseUrl, getDataImageUrl } from '@/api/request1'

export default {
  data() {
    return {
      list: null,
      listLoading: true,
      dataImageUrl: "",
      srcDataImageList: [],
      selectedModel: null,
      modelOptions:[],
      formData: {},
      active: 0,
      loadResult: false,
      inferImage: "",
      visMessageVisible:true,
      action: getApiBaseUrl() + '/interpretation/analyseAndVisualizeData',
      visMessage:{
        dataCreateTime: "",
        modelName: "",
        resultCreateTime: "",
        dataImageUrl: "",
      },
      dataResultsList: [],
      form:{},
      visContent: [
        {
          key: 0,
          label: '原图'
        },
        {
          key: 1,
          label: `船舶检测`,
        },
        {
          key: 2,
          label: `海陆分割`,
        },
        {
          key: 3,
          label: `航向预测`,
        },
      ],
      selectedVisContent: [],
      apiUrl:{
        getVisMessage: 'interpretation/getVisMessage',
        getModelList: 'model/list',
        getFileURLPictures: 'interpretation/getFileURLPictures'
      }
    }
  },
  created() {
    this.fetchVisMessage()
    this.loadModelOptions()
  },
  methods: {
    onChangeUpload(file, fileList) {
      console.log(URL.createObjectURL(file.raw))
      this.visMessage.dataImageUrl = URL.createObjectURL(file.raw);
    },
    onSuccessUpload(res, file, fileList){
      console.log(res)
      const arr = res.result.visPics[0].save_path.split("/")
      const fileName = arr[arr.length - 1]
      getActionBlob(this.apiUrl.getFileURLPictures + "/" + fileName).then((res)=>{
        let blob = new window.Blob([res.data])
        this.inferImage = window.URL.createObjectURL(blob)
      })

      // 提出数值结果
      let i = 0
      const dataResults = res.result.anaData[0]
      const dataResultColors = res.result.visPics[0].box_color
      while(i<dataResults.length){
        this.dataResultsList.push({
          'dataResults': dataResults[i] ,
          'dataResultColors': dataResultColors[i]
        })
        i++
      }
      console.log(this.dataResultsList)
      this.loadResult = false
      this.active++
    },
    fetchVisMessage(){
      postAction(this.apiUrl.getVisMessage + '/' + this.$route.query.id).then((res)=>{
        console.log(res.data.result)
        this.visMessage = res.data.result

        let arr = this.visMessage.dataImageUrl.split("\\")
        this.visMessage.dataImageUrl = getDataImageUrl(arr[arr.length-1])
        this.srcDataImageList = [this.visMessage.dataImageUrl]
      })
    },
    loadModelOptions(){
      getAction(this.apiUrl.getModelList).then((res)=>{
        console.log(res.data.result)
        this.modelOptions = res.data.result.records
      })
    },
    next(){
      if (this.active === 0){
        this.visMessageVisible = false
        this.loadResult = true
        // this.timer = setTimeout(()=>{   //TODO:设置延迟执行,如果模型算法库可用则记得去掉
        //   this.loadResult = false
        //   this.inferImage = getDataImageUrl("inference.png")
        //   console.log(this.inferImage)
        //   this.active++
        // },3000);
        this.formData.modelId = this.selectedModel
        this.formData.visOptions = this.selectedVisContent
        console.log('upload')
        this.$refs.upload.submit()
      }
      this.active++
    },
  }
}
</script>
