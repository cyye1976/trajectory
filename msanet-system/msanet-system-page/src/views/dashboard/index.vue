
<template>
  <div class="dashboard-container">
    <el-row   type="flex"  justify="center" align="middle">
      <div style="padding-right: 20px">
        <el-row   type="flex"  justify="center" align="middle">
          <el-card class="box-card" shadow="hover" style="width: 500px;">
            <div slot="header" class="clearfix">全景海上船舶态势感知系统</div>
            <div class="text item" style="width: 100%">
              基于多任务学习的遥感图像全景海上船舶态势感知系统结合了当前热门的深度学习和图像处理技术，
              通过对遥感图像数据进行分析解译，最终实现海上船舶态势可视化功能。
              本系统针对目前国内航海交通数据体量大且复杂导致的人工监控困难的问题，
              通过本系统来为航海指挥人员提供辅助决策功能，进一步保障数据分析的准确性及时效性，
              进而起到降低指挥人员决策失误率的作用，解决指挥人员由于大量的疲劳作业造成工作失误的问题。
            </div>
          </el-card>
        </el-row>
        <el-row style="padding-top: 20px"  type="flex"  justify="center" align="middle">
          <div style="padding-right: 24px;">
            <el-card class="box-card" shadow="hover" style="width: 150px;">
              <div slot="header" class="clearfix" style="text-align: center">当前模型总量</div>
              <div class="text item" style="font-size: 80px; text-align: center">
                {{ this.monitorData.modelCount }}
              </div>
            </el-card>
          </div>
          <div style="padding-right: 24px;">
            <el-card class="box-card" shadow="hover" style="width: 150px;">
              <div slot="header" class="clearfix" style="text-align: center">当前数据总量</div>
              <div class="text item" style="font-size: 80px; text-align: center">
                {{ this.monitorData.dataCount }}
              </div>
            </el-card>
          </div>

          <div>
            <el-card class="box-card" shadow="hover" style="width: 150px;">
              <div slot="header" class="clearfix" style="text-align: center">当前解译总量</div>
              <div class="text item" style="font-size: 80px; text-align: center">
                {{ this.monitorData.interpretationCount }}
              </div>
            </el-card>
          </div>
        </el-row>
        <el-row  style="padding-top: 20px"  type="flex"  justify="center" align="middle">
          <div style="padding-right: 24px;">
            <el-card class="box-card" shadow="hover" style="width: 150px;">
              <div slot="header" class="clearfix" style="text-align: center">今日模型新增</div>
              <div class="text item" style="font-size: 80px; text-align: center">
                {{ this.monitorData.modelCountCurrentDay }}
              </div>
            </el-card>
          </div>
          <div style="padding-right: 24px;">
            <el-card class="box-card" shadow="hover" style="width: 150px;">
              <div slot="header" class="clearfix" style="text-align: center">今日数据新增</div>
              <div class="text item" style="font-size: 80px; text-align: center">
                {{ this.monitorData.dataCountCurrentDay }}
              </div>
            </el-card>
          </div>

          <div>
            <el-card class="box-card" shadow="hover" style="width: 150px;">
              <div slot="header" class="clearfix" style="text-align: center">今日解译新增</div>
              <div class="text item" style="font-size: 80px; text-align: center">
                {{ this.monitorData.interpretationCountCurrentDay }}
              </div>
            </el-card>
          </div>
        </el-row>
      </div>
      <div>
        <el-row  type="flex"  justify="center" align="middle">
          <el-card class="box-card" shadow="hover" style="width: 500px;">
            <div slot="header" class="clearfix">模型利用比例</div>
            <div ref="myChart" style="width: 100%;height:300px;">

            </div>
          </el-card>
        </el-row>
        <el-row style="padding-top: 20px"  type="flex"  justify="center" align="middle">
          <el-card class="box-card" shadow="hover" style="width: 500px;">
            <div slot="header" class="clearfix">近7天已监控船舶目标数量</div>
            <div ref="ship7CountChart" style="width: 100%;height:300px;">

            </div>
          </el-card>
        </el-row>
      </div>
    </el-row>




  </div>
</template>

<script>

import { getAction } from '@/api/manage'
import echarts from 'echarts'


export default {
  name: 'Dashboard',
  data(){
    return {
      monitorData: {},
      apiUrl: {
        getAllMonitorData: 'monitor/getAllMonitorData'
      }
    }
  },
  created() {
    getAction(this.apiUrl.getAllMonitorData).then((res)=>{
      console.log(res.data)
      this.monitorData = res.data.result
      this.myCharts()
      this.ship7CountCharts()
    })
  },
  methods: {
    myCharts(){
      // 基于准备好的dom，初始化echarts实例
      let myChart = this.$echarts.init(this.$refs.myChart)
      // 绘制图表
      let option = {
        series: [
          {
            title: {
              text: '饼图'
            },
            type: 'pie',
            stillShowZeroSum: false,
            data: this.monitorData.modelUtilization5Count
          }
        ]
      };
      myChart.setOption(option);
      this.myCharts = myChart
    },
    ship7CountCharts(){
      let ship7CountChart = this.$echarts.init(this.$refs.ship7CountChart)
      getAction()
      let option = {
        xAxis: {
          data: this.monitorData.ship7count.days
        },
        yAxis: {},
        series: [
          {
            data: this.monitorData.ship7count.counts,
            type: 'line',
            label: {
              show: true,
              position: 'top',
              textStyle: {
                fontSize: 10
              }
            }
          }
        ]
      }
      ship7CountChart.setOption(option)
    }
  }
}
</script>

<style lang="scss" scoped>
.dashboard {
  &-container {
    margin: 30px;
  }
  &-text {
    font-size: 30px;
    line-height: 46px;
  }
}
</style>
