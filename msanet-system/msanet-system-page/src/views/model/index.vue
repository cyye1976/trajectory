<template>
  <div class="app-container">
    <div style="padding-bottom: 10px">
      <el-button type="primary" @click="addDialogVisible = true">上传模型</el-button>
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
      <el-table-column align="center" prop="created_at" label="操作" width="200">
        <template slot-scope="scope">
          <el-button type="success" icon="el-icon-edit" @click="editOnClick(scope.row)" />
          <el-button type="danger" icon="el-icon-delete" @click="deleteOnClick(scope.row.id)" />
        </template>
      </el-table-column>
    </el-table>
    <el-dialog
      title="编辑"
      :visible.sync="centerDialogVisible"
    >
      <span>
        <el-form ref="form" :model="form" width="80px">
          <el-form-item label="名称:">
            <el-input v-model="form.name" />
          </el-form-item>
          <el-form-item label="文件路径:">
            <el-input v-model="form.fileUrl" disabled />
          </el-form-item>
          <el-form-item label="描述:">
            <el-input v-model="form.description" type="textarea" />
          </el-form-item>
        </el-form>
      </span>
      <span slot="footer" class="dialog-footer">
        <el-button @click="centerDialogVisible = false">取 消</el-button>
        <el-button type="primary" @click="editConfirmOnClick()">确 定</el-button>
      </span>
    </el-dialog>
    <el-dialog
      title="上传模型"
      :visible.sync="addDialogVisible"
    >
      <span>
        <el-form ref="form" :model="form" width="80px" action="">
          <el-form-item label="名称:">
            <el-input v-model="form.name" />
          </el-form-item>
          <el-form-item label="文件路径:">
            <el-upload
              ref="upload"
              class="upload-demo"
              action="http://localhost:8081/model/add"
              multiple
              :limit="1"
              :data="form"
              :auto-upload="false"
              :on-success="onUploadSuccess"
              :on-error="onUploadError"
            >
              <el-button slot="trigger" size="small" type="primary">选取文件</el-button>
              <!--              <el-button style="margin-left: 10px;" size="small" type="success" @click="submitUpload">上传到服务器</el-button>-->
              <div slot="tip" class="el-upload__tip">只能上传pdparams文件，且不超过500MB</div>
            </el-upload>
          </el-form-item>
          <el-form-item label="描述:">
            <el-input v-model="form.description" type="textarea" />
          </el-form-item>
        </el-form>
      </span>
      <span slot="footer" class="dialog-footer">
        <el-button @click="addDialogVisible = false">取 消</el-button>
        <el-button type="primary" @click="submitUpload()">确 定</el-button>
      </span>
    </el-dialog>
  </div>
</template>

<script>

import { getAction, postAction } from '@/api/manage'

export default {
  data() {
    return {
      list: null,
      listLoading: false,
      centerDialogVisible: false,
      addDialogVisible: false,
      apiUrl: {
        list: '/model/list',
        delete: '/model/delete',
        update: '/model/update',
        add: '/model/add'
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
      this.listLoading = true
      getAction(this.apiUrl.delete + '/' + e).then((res) => {
        console.log(res.data)
        this.listLoading = false
        this.fetchData()
      })
    },
    editOnClick(e) {
      this.form = e
      console.log(e)
      this.centerDialogVisible = true
    },
    editConfirmOnClick() {
      postAction(this.apiUrl.update, this.form).then((res) => {
        console.log(res.data)
        this.fetchData()
        this.centerDialogVisible = false
      })
    },
    submitUpload() {
      this.$refs.upload.submit()
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
    }
  }
}
</script>
