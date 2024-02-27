<template>
  <div class="app-container">
    <div style="padding-bottom: 10px">
      <el-button type="primary" @click="addDialogVisible = true">上传数据</el-button>
    </div>
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
      <el-table-column label="描述" align="center">
        <template slot-scope="scope">
          {{ scope.row.description }}
        </template>
      </el-table-column>
      <el-table-column align="center" label="创建时间" width="200">
        <template slot-scope="scope">
          <i class="el-icon-time"/>
          <span>{{ scope.row.createTime }}</span>
        </template>
      </el-table-column>
      <el-table-column align="center" label="操作" width="270">
        <template slot-scope="scope">
          <el-button type="primary" icon="el-icon-picture-outline" @click="dataImageClick(scope.row)"></el-button>
          <el-button type="success" icon="el-icon-edit" @click="editOnClick(scope.row)"></el-button>
          <el-button type="danger" icon="el-icon-delete" @click="deleteOnClick(scope.row.id)"></el-button>
        </template>
      </el-table-column>
    </el-table>
    <el-dialog
      title="编辑"
      :visible.sync="editDialogVisible"
    >
      <span>
        <el-form ref="form" :model="form" width="80px">
          <el-form-item label="文件路径:">
            <el-input v-model="form.fileUrl" disabled></el-input>
          </el-form-item>
          <el-form-item label="描述:">
            <el-input v-model="form.description" type="textarea"></el-input>
          </el-form-item>
        </el-form>
      </span>
      <span slot="footer" class="dialog-footer">
        <el-button @click="editDialogVisible = false">取 消</el-button>
        <el-button type="primary" @click="editConfirmOnClick()">确 定</el-button>
      </span>
    </el-dialog>
    <el-dialog
      title="上传数据"
      :visible.sync="addDialogVisible"
    >
      <span>
        <el-form ref="form" :model="form" width="80px" action="">
          <el-form-item label="文件路径:">
            <el-upload
              ref="upload"
              :http-request="httpRequest"
              :on-change="onChange"
              class="upload-demo"
              :action="action"
              multiple
              :data="form"
              :auto-upload="false"
              :on-success="onUploadSuccess"
              :on-error="onUploadError"
            >
              <el-button slot="trigger" size="small" type="primary">选取文件</el-button>
              <!--              <el-button style="margin-left: 10px;" size="small" type="success" @click="submitUpload">上传到服务器</el-button>-->
              <div slot="tip" class="el-upload__tip">只能上传jpg,png,bmp文件，且不超过10MB</div>
            </el-upload>
          </el-form-item>
          <el-form-item label="描述:">
            <el-input v-model="form.description" type="textarea"></el-input>
          </el-form-item>
        </el-form>
      </span>
      <span slot="footer" class="dialog-footer">
        <el-button @click="addDialogVisible = false">取 消</el-button>
        <el-button type="primary" @click="submitUpload()">确 定</el-button>
      </span>
    </el-dialog>
    <el-dialog
      title="图片"
      :visible.sync="imageDialogVisible"
      width="30%"
      center
    >
      <div style="text-align: center">
        <el-image
          style="width: 200px; height: 200px"
          :src="dataImageUrl"
          :preview-src-list="srcDataImageList"
        >
        </el-image>
<!--        <img :src="dataImageUrl" alt="" style="width: 100%;">-->
      </div>
      <span slot="footer" class="dialog-footer">
    <el-button type="primary" @click="imageDialogVisible = false">确 定</el-button>
  </span>
    </el-dialog>
  </div>
</template>

<script>
import { getAction, getActionBlob, postAction } from '@/api/manage'
import { getApiBaseUrl, getDataImageUrl } from '@/api/request1'

export default {
  data() {
    return {
      list: null,
      listLoading: true,
      editDialogVisible: false,
      addDialogVisible: false,
      imageDialogVisible: false,
      action: getApiBaseUrl() + '/data/add',
      dataImageUrl: "",
      srcDataImageList: [],
      apiUrl: {
        list: '/data/list',
        delete: '/data/delete',
        update: '/data/update',
        add: '/data/add',
        uploadData: getApiBaseUrl() + '/data/add',
        getPicture: '/data/getPicture'
      },
      form: {}
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
    deleteOnClick(e) {
      postAction(this.apiUrl.delete + '/' + e).then((res) => {
        console.log(res.data)
        this.fetchData()
      })
    },
    editOnClick(e) {
      this.form = e
      this.editDialogVisible = true
    },
    editConfirmOnClick() {
      postAction(this.apiUrl.update, this.form).then((res) => {
        console.log(res.data)
        this.fetchData()
        this.editDialogVisible = false
      })
    },
    submitUpload() {
      console.log('upload')
      this.$refs.upload.submit()
      let param = new FormData()
      this.fileList.forEach(file => {
        param.append("files", file.raw)
        param.append("fileNames", file.name)
      });
      param.set("description", this.form.description)
      postAction(this.apiUrl.uploadData, param)
        .then(res => {
          console.log('upload success')
          this.addDialogVisible = false
          this.fetchData()
        })
    },
    dataImageClick(e) {
      // console.log(e.fileUrl)
      // const arr = e.fileUrl.split('\\')
      // const fileName = arr[arr.length - 1]
      // this.dataImageUrl = getDataImageUrl(fileName)
      getActionBlob(this.apiUrl.getPicture + "/" + e.id).then((res)=>{
        let blob = new window.Blob([res.data])
        let url = window.URL.createObjectURL(blob)
        this.dataImageUrl = url
        this.srcDataImageList = [this.dataImageUrl]
      })
      this.imageDialogVisible = true
    },
    onUploadSuccess() {
      console.log('upload success')
      this.addDialogVisible = false
      this.fetchData()
    },
    onUploadError() {
      console.log('upload error')
      this.addDialogVisible = false
      this.fetchData()
    },
    httpRequest(file) {
      console.log("httpRequest", file)
    },
    onChange(file, fileList) {
      this.fileList = fileList
    },
  }
}
</script>
